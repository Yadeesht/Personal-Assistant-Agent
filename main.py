import asyncio
import threading
import time
from datetime import datetime

import aiosqlite
import langchain_core.tools.base
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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
)

# Import tools 
import app_tools.tools.google.gmail_tools
import app_tools.tools.google.calendar_tools
import app_tools.tools.google.gsheet_tools
import app_tools.tools.google.gdocs_tools
import app_tools.tools.google.gtask_tools

from config.settings import CHECKPOINT_DB, DEFAULT_THREAD_ID
from core.graph import build_graph
from rich.markdown import Markdown
from rich.panel import Panel
from utils.helper import console, request_counter, setup_logger

logger = setup_logger(__name__)


AGENT_MESSAGE_KEY = {
    "supervisor": "supervisor_messages",
    "communication_agent": "communication_messages",
    "planning_agent": "planning_messages",
    "document_agent": "document_messages",
    "data_agent": "data_messages",
}


def next_thread_id(current_id: str) -> str:
    """Increment thread ID: '7' -> '8', 'thread_1' -> 'thread_2', etc."""
    current_id = str(current_id).strip()
    if current_id.isdigit():
        return str(int(current_id) + 1)
    import re

    match = re.search(r"(\d+)$", current_id)
    if match:
        num = int(match.group(1)) + 1
        prefix = current_id[: match.start(1)]
        return f"{prefix}{num}"
    return f"{current_id}_2"


def update_env_thread_id(new_thread_id: str):
    """Persist the new DEFAULT_THREAD_ID to .env file."""
    try:
        import dotenv
        from config.settings import BASE_DIR

        env_path = BASE_DIR / ".env"
        if env_path.exists():
            dotenv.set_key(
                env_path, "DEFAULT_THREAD_ID", str(new_thread_id), quote_mode="never"
            )
    except Exception as e:
        logger.warning(f"Could not persist DEFAULT_THREAD_ID to .env: {e}")


def reload_prompts_and_graph(tool_sets, checkpointer):
    """Hot-reload prompt definitions and rebuild graph in memory."""
    import importlib

    prompts_mod = importlib.import_module("config.prompts")
    graph_mod = importlib.import_module("core.graph")
    importlib.reload(prompts_mod)
    importlib.reload(graph_mod)
    return graph_mod.build_graph(tool_sets, checkpointer)


def keyword_listener(
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    agent_state,
    turn_ready: threading.Event,
):
    """Read stdin on a dedicated daemon thread."""
    while True:
        turn_ready.wait()
        try:
            console.print("\n[bold #D97757]You[/] [bold #D97757]❯[/] ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            try:
                asyncio.run_coroutine_threadsafe(
                    queue.put(("TEXT", "exit")), loop
                ).result()
            except Exception:
                pass
            break
        except Exception as e:
            logger.error(f"Error in keyword listener: {e}")
            continue

        if not user_input.strip():
            continue

        turn_ready.clear()
        agent_state["last_interaction"] = time.time()
        try:
            asyncio.run_coroutine_threadsafe(
                queue.put(("TEXT", user_input.strip())), loop
            ).result()
        except Exception as e:
            logger.error(f"Failed to enqueue user input: {e}")


async def main():
    start_time = datetime.now()
    console.rule("[bold #D97757]Personal Assistant Agent[/]")
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

        tool_sets = {
            "communication": communication_tools,
            "planning": planning_tools,
            "content": content_tools,
        }

        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(CHECKPOINT_DB)) as connection:
            checkpointer = AsyncSqliteSaver(connection)
            graph = build_graph(tool_sets, checkpointer)

            current_thread_id = str(DEFAULT_THREAD_ID)
            # LangGraph's default recursion limit (25 steps) ends long multi-app tasks with an error;
            # 310 matches the eval (3 graph steps per model call x 100 calls + 10).
            run_config = {
                "configurable": {
                    "thread_id": current_thread_id,
                },
                "recursion_limit": 310,
            }

            agent_state = {"last_interaction": 0}

            event_queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            turn_ready = threading.Event()

            threading.Thread(
                target=keyword_listener,
                args=(event_queue, loop, agent_state, turn_ready),
                daemon=True,
            ).start()

            console.print(
                f"[dim]Active Thread: [bold cyan]{current_thread_id}[/bold cyan] • "
                f"Commands: [bold cyan]/new[/bold cyan] (fresh chat), [bold cyan]/reload[/bold cyan] (update prompts), [bold cyan]/clear[/bold cyan], [bold cyan]/exit[/bold cyan][/dim]"
            )
            turn_ready.set()

            state = {"messages": []}

            while True:
                _, query = await event_queue.get()

                agent_state["last_interaction"] = time.time()
                query_clean = query.strip()
                cmd_parts = query_clean.split(maxsplit=1)
                cmd = cmd_parts[0].lower() if cmd_parts else ""
                cmd_arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

                if cmd in ["exit", "quit", "bye", "/exit", "/quit"]:
                    console.print("\n[dim]👋 Goodbye![/dim]\n")
                    break

                if cmd in ["clear", "/clear", "cls"]:
                    console.clear()
                    console.rule("[bold #D97757]Personal Assistant Agent[/]")
                    turn_ready.set()
                    continue

                if cmd in ["/new", "/reset", "new", "reset"]:
                    current_thread_id = next_thread_id(current_thread_id)
                    run_config["configurable"]["thread_id"] = current_thread_id
                    update_env_thread_id(current_thread_id)

                    # Hot-reload prompt definitions and rebuild graph
                    graph = reload_prompts_and_graph(tool_sets, checkpointer)

                    state = {"messages": []}
                    console.clear()
                    console.rule("[bold #D97757]Personal Assistant Agent[/]")
                    console.print(
                        f"[bold green]✨ Started fresh chat session![/bold green] "
                        f"[dim](Thread: [bold cyan]{current_thread_id}[/bold cyan] saved to .env • Prompts reloaded)[/dim]\n"
                    )
                    turn_ready.set()
                    continue

                if cmd in ["/reload", "reload"]:
                    graph = reload_prompts_and_graph(tool_sets, checkpointer)

                    console.print(
                        f"[bold green]🔄 Prompts reloaded successfully![/bold green] "
                        f"[dim](Graph rebuilt in thread: [bold cyan]{current_thread_id}[/bold cyan])[/dim]\n"
                    )
                    turn_ready.set()
                    continue

                if cmd in ["/thread", "thread"]:
                    if cmd_arg:
                        current_thread_id = cmd_arg
                        run_config["configurable"]["thread_id"] = current_thread_id
                        update_env_thread_id(current_thread_id)
                        state = {"messages": []}
                        console.print(
                            f"[bold green]Switched to thread:[/bold green] [bold cyan]{current_thread_id}[/bold cyan] [dim](saved to .env)[/dim]\n"
                        )
                    else:
                        console.print(
                            f"[dim]Current active thread:[/dim] [bold cyan]{current_thread_id}[/bold cyan]\n"
                        )
                    turn_ready.set()
                    continue

                if cmd in ["/help", "help"]:
                    console.print(
                        Panel(
                            "[bold cyan]/new[/bold cyan] or [bold cyan]/reset[/bold cyan]   — Start a fresh chat (increments thread ID & hot-reloads prompts)\n"
                            "[bold cyan]/reload[/bold cyan]          — Reload prompts & rebuild graph in the current chat without restarting\n"
                            "[bold cyan]/clear[/bold cyan]           — Clear the screen\n"
                            "[bold cyan]/thread [id][/bold cyan]     — Show active thread ID or switch to a specific thread\n"
                            "[bold cyan]/exit[/bold cyan]            — Quit session",
                            title="[bold #D97757]Available Commands[/bold #D97757]",
                            border_style="#D97757",
                            padding=(0, 2),
                        )
                    )
                    turn_ready.set()
                    continue

                request_counter.start_turn(query)
                snapshot = await graph.aget_state(run_config)
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

                with console.status("[bold #D97757]Thinking...[/]", spinner="dots"):
                    state = await graph.ainvoke(new_input, config=run_config)
                request_counter.end_turn()

                active_agent = state.get("current_agent", current_agent)
                response_key = AGENT_MESSAGE_KEY.get(
                    active_agent, "supervisor_messages"
                )

                messages = state.get(response_key) or state.get("messages", [])
                last_msg = messages[-1] if messages else None

                if isinstance(last_msg, AIMessage) and last_msg.content:
                    final_response = last_msg.content
                    console.print()
                    console.print(
                        Panel(
                            Markdown(final_response),
                            title="[bold #D97757]Assistant[/]",
                            border_style="#D97757",
                            padding=(1, 2),
                        )
                    )
                    agent_state["last_interaction"] = time.time()

                turn_ready.set()

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            console.print()
            console.print(
                Panel(
                    f"• LLM requests: [cyan]{request_counter.session_total()}[/]\n"
                    f"• Messages: [cyan]{len(state['messages'])}[/]\n"
                    f"• Elapsed Time: [cyan]{execution_time:.2f}s[/]",
                    title="[bold]Session Complete[/bold]",
                    border_style="dim",
                    padding=(0, 2),
                )
            )

    except Exception as e:
        logger.exception(f"❌ An error occurred: {e}")
        raise e


if __name__ == "__main__":
    asyncio.run(main())
