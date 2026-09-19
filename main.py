import os

# Disable TensorFlow backend detection in Hugging Face Transformers
# (this repository only uses PyTorch for embeddings and avoids Keras 3 conflicts)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import asyncio
import threading
import time
from datetime import datetime

import aiosqlite
import langchain_core.tools.base
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

if not hasattr(langchain_core.tools.base, "TOOL_MESSAGE_BLOCK_TYPES"):
    langchain_core.tools.base.TOOL_MESSAGE_BLOCK_TYPES = (
        "text",
        "image_url",
        "image",
        "json",
        "search_result",
        "custom_tool_call_output",
        "document",
        "file",
    )

from app_tools.core.server_init import (
    communication_server,
    planning_server,
    content_server,
    supervisor_server,
)

# Import tools 
import app_tools.tools.google.gmail_tools
import app_tools.tools.google.calendar_tools
import app_tools.tools.google.gdrive_tools
import app_tools.tools.google.gslide_tools
import app_tools.tools.google.gsheet_tools
import app_tools.tools.google.gform_tools
import app_tools.tools.google.gdocs_tools
import app_tools.tools.google.gtask_tools
import app_tools.tools.google.gsearch_tools
import app_tools.tools.rag_tools

from config.settings import CHECKPOINT_DB, DEFAULT_THREAD_ID
from core.graph import build_graph
from utils.helper import (
    AsyncSqliteSaver,
    request_counter,
    setup_logger,
)
from utils.memory_manager import log_event

logger = setup_logger(__name__)


AGENT_MESSAGE_KEY = {
    "supervisor": "supervisor_messages",
    "communication_agent": "communication_messages",
    "planning_agent": "planning_messages",
    "document_agent": "document_messages",
    "presentation_agent": "presentation_messages",
    "data_agent": "data_messages",
    "code_agent": "code_messages",
}


def keyword_listener(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, agent_state):
    """Read stdin on a dedicated daemon thread.

    `loop.run_in_executor(None, input)` schedules the blocking read on
    asyncio's default ThreadPoolExecutor, which `asyncio.run()` waits on
    (`loop.shutdown_default_executor()`) before the process can exit. That
    made the process hang after "Goodbye!" until one more Enter press
    unblocked the pending input() call. A plain daemon thread isn't awaited
    by asyncio.run(), so the process can exit immediately.
    """
    while True:
        try:
            user_input = input()
        except EOFError:
            break
        except Exception as e:
            logger.error(f"Error in keyword listener: {e}")
            continue

        if user_input.strip():
            agent_state["last_interaction"] = time.time()
            try:
                asyncio.run_coroutine_threadsafe(
                    queue.put(("TEXT", user_input.strip())), loop
                ).result()
            except Exception as e:
                logger.error(f"Failed to enqueue user input: {e}")


async def main():
    start_time = datetime.now()
    logger.info("🚀 Starting Agent")

    try:
        def get_langchain_tools(server) -> list[StructuredTool]:
            return server.list_tools()

        communication_tools = get_langchain_tools(communication_server)
        logger.info(f"📧 Communication Tools: {len(communication_tools)}")

        planning_tools = get_langchain_tools(planning_server)
        logger.info(f"✅ Planning Tools: {len(planning_tools)}")

        content_tools = get_langchain_tools(content_server)
        logger.info(f"📺 Content Tools: {len(content_tools)}")

        supervisor_tools = get_langchain_tools(supervisor_server)
        logger.info(f"🔍 Supervisor Tools: {len(supervisor_tools)}")

        tool_sets = {
            "communication": communication_tools,
            "planning": planning_tools,
            "content": content_tools,
            "supervisor": supervisor_tools,
        }

        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(CHECKPOINT_DB)) as connection:
            checkpointer = AsyncSqliteSaver(connection)
            graph = build_graph(tool_sets, checkpointer)

            # g = graph.get_graph()

            # png_bytes = g.draw_mermaid_png()

            # with open("docs/images/agent_structure_graph.png", "wb") as f:
            #     f.write(png_bytes)

            config = {
                "configurable": {
                    "thread_id": DEFAULT_THREAD_ID,
                }
            }

            agent_state = {"last_interaction": 0}

            event_queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            threading.Thread(
                target=keyword_listener,
                args=(event_queue, loop, agent_state),
                daemon=True,
            ).start()

            logger.info("⌨️ Type your message")
            logger.info("💡 Type 'exit' or 'quit' to stop\n")

            state = {"messages": []}

            while True:
                _, query = await event_queue.get()

                agent_state["last_interaction"] = time.time()

                if query.lower() in ["exit", "quit", "bye"]:
                    logger.info("👋 Goodbye!")
                    break

                logger.info(f"👤 You: {query}")

                try:
                    await log_event(
                        thread_id=DEFAULT_THREAD_ID,
                        actor="Human_node",
                        message=query,
                        metadata={},
                    )
                except Exception as e:
                    logger.error(f"Failed to log human_node audit event: {e}")

                request_counter.start_turn(query)
                snapshot = await graph.aget_state(config)
                current_agent = "supervisor"
                if snapshot and snapshot.values:
                    current_agent = snapshot.values.get("current_agent", "supervisor")

                context_key = AGENT_MESSAGE_KEY.get(
                    current_agent, "supervisor_messages"
                )
                human_message = HumanMessage(content=query)

                new_input = {
                    "messages": [human_message],
                    context_key: [human_message],
                }

                state = await graph.ainvoke(new_input, config=config)
                request_counter.end_turn()

                active_agent = state.get("current_agent", current_agent)
                response_key = AGENT_MESSAGE_KEY.get(
                    active_agent, "supervisor_messages"
                )

                messages = state.get(response_key) or state.get("messages", [])
                last_msg = messages[-1] if messages else None

                if isinstance(last_msg, AIMessage) and last_msg.content:
                    final_response = last_msg.content
                    logger.info(f"🤖 Agent: {final_response}")

                    agent_state["last_interaction"] = time.time()

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(
                f"🎯 Session complete | "
                f"LLM requests: {request_counter.session_total()} | "
                f"Messages: {len(state['messages'])} | "
                f"Time: {execution_time:.2f}s"
            )

    except Exception as e:
        logger.exception(f"❌ An error occurred: {e}")
        raise e


if __name__ == "__main__":
    asyncio.run(main())
