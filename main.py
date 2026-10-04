import os

# Disable TensorFlow backend detection in Hugging Face Transformers
# (this repository only uses PyTorch for embeddings and avoids Keras 3 conflicts)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import asyncio
import json
import threading
import time
from datetime import datetime

import aiosqlite
import langchain_core.tools.base
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

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
from core.agent import memory_saver
from core.graph import build_graph
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from utils.command_prompt import COMMANDS, interactive_terminal, make_session, read_line
from utils.helper import console, first_line, request_counter, setup_logger
from utils.memory_manager import has_pending_memory, init_memory_progress, log_event

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


async def save_memory(reason: str):
    """Start storing the conversation logs not yet in long-term memory (knowledge graph
    and episodic RAG), from every thread, on the background saver; the chat carries on.
    Runs when a chat ends (/new, /thread) and at startup for anything an earlier session
    left unsaved (e.g. closed with Ctrl+C)."""
    try:
        if await has_pending_memory():
            memory_saver.request(reason)
    except Exception as e:
        logger.error(f"Saving long-term memory failed: {e}")


async def finish_saving_memory():
    """On exit: store what is left and wait for the background saver to finish."""
    try:
        if await has_pending_memory():
            memory_saver.request("exit")
        if not memory_saver.running:
            return
        with console.status(
            "[bold #D97757]Saving to long-term memory...[/]", spinner="dots"
        ):
            await asyncio.to_thread(memory_saver.wait)
        result = memory_saver.last_result or {}
        if not all(result.values()):
            failed = ", ".join(name for name, ok in result.items() if not ok)
            console.print(
                f"[yellow]Could not save all memory ({failed}); it will be retried next time.[/yellow]"
            )
    except Exception as e:
        logger.error(f"Saving long-term memory failed: {e}")


async def pending_confirmation(graph, config):
    """The confirmation the graph is paused for in this thread (core/confirm.py), if any."""
    snapshot = await graph.aget_state(config)
    for task in getattr(snapshot, "tasks", None) or ():
        for item in getattr(task, "interrupts", None) or ():
            value = getattr(item, "value", None)
            if isinstance(value, dict) and value.get("kind") == "confirm_tool_calls":
                return value
    return None


def show_confirmation(request: dict):
    """Show exactly what would be done, and how to answer."""
    parts = []
    for call in request.get("calls", []):
        args = call.get("args") or {}
        if call.get("tool") == "send_email":
            parts.append(
                f"[bold]To:[/bold] {escape(str(args.get('recipient_id', '')))}\n"
                f"[bold]Subject:[/bold] {escape(str(args.get('subject', '')))}\n\n"
                f"{escape(str(args.get('message', '')))}"
            )
        else:
            parts.append(
                f"[bold]{escape(str(call.get('tool')))}[/bold]\n"
                f"{escape(json.dumps(args, indent=2, default=str))}"
            )
    console.print()
    console.print(
        Panel(
            "\n\n[dim]───[/dim]\n\n".join(parts),
            title="[bold yellow]Confirm before sending[/]",
            subtitle="[dim]Type [bold]yes[/bold] to send · anything else cancels and is passed to the agent[/dim]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def show_reply(state: dict, fallback_agent: str):
    """Print the active agent's latest reply, if it has one."""
    active_agent = state.get("current_agent", fallback_agent)
    response_key = AGENT_MESSAGE_KEY.get(active_agent, "supervisor_messages")
    messages = state.get(response_key) or state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if isinstance(last_msg, AIMessage) and last_msg.content:
        console.print()
        console.print(
            Panel(
                Markdown(last_msg.content),
                title="[bold #D97757]Assistant[/]",
                border_style="#D97757",
                padding=(1, 2),
            )
        )


def reload_prompts_and_graph(tool_sets, checkpointer):
    """Re-read config/prompts.py and rebuild the graph with it.

    Only the prompt text is reloaded: the graph reads the prompts from that module when
    it is built. Code modules are not reloaded (mixing reloaded and already-loaded code
    breaks imports); code changes need a restart.
    """
    import importlib

    import config.prompts

    importlib.reload(config.prompts)
    return build_graph(tool_sets, checkpointer)


def try_reload_prompts(tool_sets, checkpointer, graph):
    """Reload the prompts; on failure keep the current graph. Returns (graph, error)."""
    try:
        return reload_prompts_and_graph(tool_sets, checkpointer), None
    except Exception as e:
        logger.error(f"Reloading prompts failed: {e}")
        return graph, e


def keyword_listener(
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    agent_state,
    turn_ready: threading.Event,
):
    """Read stdin on a dedicated daemon thread.

    In a terminal the input line pops up the slash commands as you type
    (utils/command_prompt.py); otherwise (piped input, tests) plain input() is used.
    """
    session = make_session() if interactive_terminal() else None
    while True:
        turn_ready.wait()
        try:
            if session is not None:
                console.print()
                user_input = read_line(session)
            else:
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

            current_thread_id = str(DEFAULT_THREAD_ID)
            # LangGraph's default recursion limit (25 steps) ends long multi-app tasks with an error;
            # 310 matches the eval (3 graph steps per model call x 100 calls + 10).
            run_config = {
                "configurable": {
                    "thread_id": current_thread_id,
                },
                "recursion_limit": 310,
            }

            # Memory progress used to be a per-thread timestamp in graph state; carry it
            # over once so upgrading does not store the same logs twice.
            snapshot = await graph.aget_state(run_config)
            await init_memory_progress(
                (snapshot.values or {}).get("last_memory_timestamp") if snapshot else None
            )
            await save_memory("startup")

            # Load the memory embedding models while the user types the first message,
            # so recall does not have to wait for them.
            threading.Thread(
                target=app_tools.tools.rag_tools.preload_memory_models,
                name="memory-preload",
                daemon=True,
            ).start()

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
            # A send can still be waiting for the user's answer from an earlier session.
            confirmation = await pending_confirmation(graph, run_config)
            if confirmation:
                show_confirmation(confirmation)
            turn_ready.set()

            state = {"messages": []}

            while True:
                _, query = await event_queue.get()
                try:

                    agent_state["last_interaction"] = time.time()
                    query_clean = query.strip()
                    cmd_parts = query_clean.split(maxsplit=1)
                    cmd = cmd_parts[0].lower() if cmd_parts else ""
                    cmd_arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

                    if cmd in ["exit", "quit", "bye", "/exit", "/quit"]:
                        await finish_saving_memory()
                        console.print("\n[dim]👋 Goodbye![/dim]\n")
                        break

                    if cmd in ["clear", "/clear", "cls"]:
                        console.clear()
                        console.rule("[bold #D97757]Personal Assistant Agent[/]")
                        turn_ready.set()
                        continue

                    if cmd in ["/new", "/reset", "new", "reset"]:
                        await save_memory(f"leaving thread {current_thread_id}")
                        current_thread_id = next_thread_id(current_thread_id)
                        run_config["configurable"]["thread_id"] = current_thread_id
                        update_env_thread_id(current_thread_id)

                        # Re-read config/prompts.py for the fresh chat (keeps the old ones on failure)
                        graph, reload_error = try_reload_prompts(tool_sets, checkpointer, graph)

                        state = {"messages": []}
                        confirmation = None
                        console.clear()
                        console.rule("[bold #D97757]Personal Assistant Agent[/]")
                        console.print(
                            f"[bold green]✨ Started fresh chat session![/bold green] "
                            f"[dim](Thread: [bold cyan]{current_thread_id}[/bold cyan] saved to .env • "
                            f"{'Prompts reloaded' if reload_error is None else 'Prompts unchanged'})[/dim]\n"
                        )
                        if reload_error is not None:
                            console.print(
                                f"[yellow]Could not reload config/prompts.py: {escape(first_line(reload_error))}. "
                                "Kept the previous prompts.[/yellow]\n"
                            )
                        turn_ready.set()
                        continue

                    if cmd in ["/reload", "reload"]:
                        graph, reload_error = try_reload_prompts(tool_sets, checkpointer, graph)
                        if reload_error is None:
                            console.print(
                                f"[bold green]🔄 Prompts reloaded successfully![/bold green] "
                                f"[dim](Graph rebuilt in thread: [bold cyan]{current_thread_id}[/bold cyan])[/dim]\n"
                            )
                        else:
                            console.print(
                                f"[red]Could not reload config/prompts.py: {escape(first_line(reload_error))}[/red]\n"
                                "[dim]Kept the previous prompts. Fix the file and run /reload again.[/dim]\n"
                            )
                        turn_ready.set()
                        continue

                    if cmd in ["/thread", "thread"]:
                        if cmd_arg:
                            await save_memory(f"leaving thread {current_thread_id}")
                            current_thread_id = cmd_arg
                            run_config["configurable"]["thread_id"] = current_thread_id
                            update_env_thread_id(current_thread_id)
                            state = {"messages": []}
                            console.print(
                                f"[bold green]Switched to thread:[/bold green] [bold cyan]{current_thread_id}[/bold cyan] [dim](saved to .env)[/dim]\n"
                            )
                            confirmation = await pending_confirmation(graph, run_config)
                            if confirmation:
                                show_confirmation(confirmation)
                        else:
                            console.print(
                                f"[dim]Current active thread:[/dim] [bold cyan]{current_thread_id}[/bold cyan]\n"
                            )
                        turn_ready.set()
                        continue

                    if cmd in ["/help", "help"]:
                        console.print(
                            Panel(
                                "\n".join(
                                    f"[bold cyan]{command:<9}[/bold cyan] {description}"
                                    for command, description in COMMANDS.items()
                                )
                                + "\n\n[dim]Tip: type / to pick a command from the pop-up list.[/dim]",
                                title="[bold #D97757]Available Commands[/bold #D97757]",
                                border_style="#D97757",
                                padding=(0, 2),
                            )
                        )
                        turn_ready.set()
                        continue

                    if cmd.startswith("/"):
                        # A mistyped command is not sent to the assistant as a message.
                        console.print(
                            f"[yellow]Unknown command {escape(cmd)}.[/yellow] "
                            "[dim]Type / to see the commands.[/dim]"
                        )
                        turn_ready.set()
                        continue

                    try:
                        await log_event(
                            thread_id=current_thread_id,
                            actor="Human_node",
                            message=query,
                            metadata={},
                        )
                    except Exception as e:
                        logger.error(f"Failed to log human_node audit event: {e}")

                    request_counter.start_turn(query)
                    snapshot = await graph.aget_state(run_config)
                    current_agent = "supervisor"
                    if snapshot and snapshot.values:
                        current_agent = snapshot.values.get("current_agent", "supervisor")

                    if confirmation:
                        # This message answers the confirmation: resume the paused step. The
                        # tool node sends only if the reply is a yes (core/confirm.py).
                        with console.status("[bold #D97757]Thinking...[/]", spinner="dots"):
                            state = await graph.ainvoke(
                                Command(resume={"reply": query}), config=run_config
                            )
                    else:
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

                    show_reply(state, current_agent)
                    agent_state["last_interaction"] = time.time()

                    confirmation = await pending_confirmation(graph, run_config)
                    if confirmation:
                        show_confirmation(confirmation)

                    turn_ready.set()
                except Exception as e:
                    # One failing message or command does not end the session.
                    logger.exception(f"That message failed: {e}")
                    console.print(
                        Panel(
                            f"[red]{escape(first_line(e))}[/red]\n"
                            "[dim]Nothing else was affected; you can keep chatting. "
                            "If it keeps happening, restart the app.[/dim]",
                            title="[bold red]Something went wrong[/]",
                            border_style="red",
                        )
                    )
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
