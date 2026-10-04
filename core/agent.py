import asyncio
import json
import threading
import time
from datetime import datetime

import aiosqlite
import httpx
from langchain_core.messages import (
    AIMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)

# Read at call time, so a /reload of config/prompts.py applies here too.
import config.prompts as prompt_texts
from config.settings import DEFAULT_THREAD_ID, MEMORY_DB
from core.codeagent import CodeExecutionAgent
from core.llm import build_llm
from core.state import State, plan_summary
from rag.episodic_rag import EpisodicRAG
from utils.helper import (
    count_tokens,
    format_tool_to_text,
    get_current_time,
    request_counter,
    sanitize_history,
    setup_logger,
)
from utils.memory_manager import (
    MEMORY_PIPELINES,
    get_memory_progress,
    log_event,
    set_memory_progress,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
import langchain_core.tools.base
from langchain_core.tools import tool, InjectedToolCallId

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

from langgraph.prebuilt import InjectedState
from typing import Annotated
from langchain_core.messages import HumanMessage

logger = setup_logger(__name__)


def clean_unmatched_tool_calls(messages: list) -> list:
    """
    Ensures that any AIMessage with tool_calls is matched by subsequent ToolMessages.
    If any tool call lacks a corresponding ToolMessage in the rest of the list,
    the tool_calls are stripped from the AIMessage to avoid OpenAI API BadRequestError.
    A ToolMessage whose tool call is not earlier in the list (e.g. the call was removed
    by the summarizer) is dropped too: the API rejects a tool result without its call.
    """
    cleaned_messages = []
    
    # Track all tool_call_ids that actually have corresponding ToolMessages in the list
    existing_tool_message_ids = {
        msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)
    }

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Check if every tool call in this AIMessage has a corresponding ToolMessage
            matched_calls = []
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id and tc_id in existing_tool_message_ids:
                    matched_calls.append(tc)
            
            if len(matched_calls) == len(msg.tool_calls):
                # All tool calls are matched, keep as is
                cleaned_messages.append(msg)
            elif len(matched_calls) > 0:
                # Partially matched: keep only the matched ones
                new_msg = AIMessage(
                    content=msg.content,
                    name=msg.name,
                    id=msg.id,
                    tool_calls=matched_calls,
                    additional_kwargs=msg.additional_kwargs.copy() if msg.additional_kwargs else {}
                )
                cleaned_messages.append(new_msg)
            else:
                # None of the tool calls are matched: strip tool_calls completely
                new_msg = AIMessage(
                    content=msg.content,
                    name=msg.name,
                    id=msg.id,
                    additional_kwargs=msg.additional_kwargs.copy() if msg.additional_kwargs else {}
                )
                cleaned_messages.append(new_msg)
        else:
            cleaned_messages.append(msg)

    answered_call_ids = set()
    result = []
    for msg in cleaned_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            answered_call_ids.update(tc.get("id") for tc in msg.tool_calls)
        if isinstance(msg, ToolMessage) and msg.tool_call_id not in answered_call_ids:
            continue
        result.append(msg)

    return result


AGENT_MESSAGE_KEY = {
    "supervisor": "supervisor_messages",
    "communication_agent": "communication_messages",
    "planning_agent": "planning_messages",
    "document_agent": "document_messages",
    "presentation_agent": "presentation_messages",
    "data_agent": "data_messages",
    "code_agent": "code_messages",
}


def _resolve_agent_messages(state: State, agent_name: str):
    message_key = AGENT_MESSAGE_KEY.get(agent_name, "messages")
    return state.get(message_key, []) or state.get("messages", [])


def _thread_id(config: RunnableConfig) -> str:
    """The thread this run belongs to, so logs and memory follow /new and /thread."""
    return (config or {}).get("configurable", {}).get("thread_id") or DEFAULT_THREAD_ID


async def _memory_for_turn(state: State) -> tuple[str, str, str, dict]:
    """Long-term memory for this turn: the user's standing instructions and profile
    (always), and knowledge-graph facts related to the user's latest message (looked
    up once per turn).

    Returns (instructions, profile, recalled text, state updates). The recalled text is
    kept in state with the id of the message it was looked up for, so the supervisor and
    the workers of the same turn reuse one lookup. Recall leaves out the entities the
    profile shows. Past conversations are not recalled here; the supervisor looks them
    up on request.
    """
    from app_tools.tools.rag_tools import get_instructions_text, get_profile, recall_memory

    instructions = await get_instructions_text()

    try:
        profile, profile_ids = await asyncio.to_thread(get_profile)
    except Exception as e:
        logger.error(f"Loading the memory profile failed: {e}")
        profile, profile_ids = "", set()

    latest = next(
        (
            m
            for m in reversed(state.get("messages", []))
            if isinstance(m, HumanMessage) and not m.name
        ),
        None,
    )
    if latest is None:
        return instructions, profile, "", {}
    if latest.id and state.get("recalled_for") == latest.id:
        return instructions, profile, state.get("recalled_memory") or "", {}

    query = latest.content if isinstance(latest.content, str) else str(latest.content)
    try:
        recalled = await asyncio.to_thread(recall_memory, query, profile_ids)
    except Exception as e:
        logger.error(f"Memory recall failed: {e}")
        recalled = ""
    if recalled:
        logger.info(f"🧠 Recalled memory for this turn:\n{recalled}")
    return (
        instructions,
        profile,
        recalled,
        {"recalled_memory": recalled, "recalled_for": latest.id},
    )


def _memory_messages(instructions: str, profile: str, recalled: str) -> list:
    """The messages placed after an agent's system prompt: the user's standing
    instructions (rules to follow), then long-term memory (facts that may be out of
    date). None for what is empty."""
    messages = []
    if instructions:
        messages.append(
            SystemMessage(content=prompt_texts.STANDING_INSTRUCTIONS_TEMPLATE.format(instructions=instructions))
        )
    sections = []
    if profile:
        sections.append(f"About the user:\n{profile}")
    if recalled:
        sections.append(recalled)
    if sections:
        messages.append(
            SystemMessage(content=prompt_texts.MEMORY_RECALL_TEMPLATE.format(memory="\n\n".join(sections)))
        )
    return messages


def _summary_messages(state: State) -> list:
    """The thread summary (older turns condensed by the summarizer), if there is one."""
    summary = state.get("summary")
    if not summary:
        return []
    return [SystemMessage(content=f"Conversation Summary of previous messages:\n{summary}")]


def supervisor_node_factory(
    llm_with_tools,
    system_prompt,
    agent_name="supervisor",
):
    """Create the supervisor graph node with isolated supervisor context."""

    async def supervisor_node(state: State, config: RunnableConfig):
        request_counter[agent_name] += 1
        request_num = request_counter[agent_name]
        thread_id = _thread_id(config)

        current_time = get_current_time()

        logger.info(f"👮 SUPERVISOR REQUEST #{request_num}")

        scoped_messages = _resolve_agent_messages(state, agent_name)
        logger.info(f"📨 Messages in supervisor context: {len(scoped_messages)}")

        # No token cap: the supervisor sees its whole history. Long threads are condensed
        # between turns by the summarizer (route_start), not trimmed here.
        last_messages = list(scoped_messages)

        logger.info("=" * 80)
        if last_messages:  # this is for logs purpose only
            content_preview = sanitize_history(last_messages)
            content_preview = json.dumps(content_preview[-2:], indent=2)
            logger.info(f"📝 Content preview: {content_preview}")

        instructions, profile, recalled, recall_updates = await _memory_for_turn(state)

        try:
            final_prompt = system_prompt.replace("{current_time}", current_time)

            message = (
                [SystemMessage(content=final_prompt)]
                + _memory_messages(instructions, profile, recalled)
                + _summary_messages(state)
                + last_messages
            )
            message = clean_unmatched_tool_calls(message)
            response = await llm_with_tools.ainvoke(message)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.error("🚫 Rate limit hit in supervisor - stopping")
                return {
                    "next": "FINISH",
                    "messages": [
                        AIMessage(
                            content="[ERROR] Rate limit reached.",
                            name=agent_name,
                            additional_kwargs={"timestamp": current_time},
                        )
                    ],
                }
            logger.error(f"HTTP error in supervisor: {e}")
            return {
                "next": "FINISH",
                "messages": [
                    AIMessage(
                        content=f"[ERROR] {e}",
                        name=agent_name,
                        additional_kwargs={"timestamp": current_time},
                    )
                ],
            }
        except httpx.RequestError as e:
            logger.error(f"🚫 Network error in supervisor: {e}")
            return {
                "next": "FINISH",
                "messages": [
                    AIMessage(
                        content="[ERROR] Network unavailable.",
                        name=agent_name,
                        additional_kwargs={"timestamp": current_time},
                    )
                ],
            }
        except Exception as e:
            logger.error(f"Error in supervisor_node: {e}")
            return {
                "messages": [
                    AIMessage(
                        content=f"[ERROR] {e}",
                        name=agent_name,
                        additional_kwargs={"timestamp": current_time},
                    ),
                ],
                "supervisor_messages": [
                    AIMessage(
                        content=f"[ERROR] {e}",
                        name=agent_name,
                        additional_kwargs={"timestamp": current_time},
                    )
                ],
                "current_agent": agent_name,
            }

        for i, tool_call in enumerate(getattr(response, "tool_calls", []), 1):
            logger.info(f"   Tool #{i}:")
            logger.info(f"      Name: {tool_call.get('name', 'N/A')}")
            logger.info(
                f"      Args: {json.dumps(tool_call.get('args', {}), indent=10)}"
            )

        agent_message = AIMessage(
            content=response.content,
            name=agent_name,
            tool_calls=getattr(response, "tool_calls", []),
            additional_kwargs={"timestamp": current_time},
        )

        try:
            has_tools = bool(getattr(response, "tool_calls", []))
            if has_tools:
                await log_event(
                    thread_id=thread_id,
                    actor=agent_name,
                    message=f"{', '.join([format_tool_to_text(tc.get('name', ''), json.dumps(tc.get('args', {}))) for tc in response.tool_calls])}",
                    metadata={"request_num": request_num, "type": "tool_call"},
                )
            elif response.content:
                await log_event(
                    thread_id=thread_id,
                    actor=agent_name,
                    message=f"Direct response: {response.content}",
                    metadata={"request_num": request_num, "type": "content"},
                )
        except Exception as e:
            logger.error(f"Failed to log supervisor event: {e}")

        return {
            "messages": [agent_message],
            "supervisor_messages": [agent_message],
            "current_agent": agent_name,
            **recall_updates,
        }

    return supervisor_node




def agent_node_factory(llm_with_tools, system_prompt, agent_name: str):
    """Create a specialized worker-agent node.

    The returned node invokes the worker LLM with isolated agent context.
    """

    async def agent_node(state: State, config: RunnableConfig):

        current_agent_name = agent_name
        request_counter[current_agent_name] += 1
        request_num = request_counter[current_agent_name]
        thread_id = _thread_id(config)

        current_time = get_current_time()

        logger.info("\n")
        logger.info("=" * 80)
        logger.info(f"🔄 {current_agent_name.upper()} REQUEST #{request_num}")
        logger.info("=" * 80)

        # No token cap: a worker sees its whole task history (a token cap starting on the
        # handoff message emptied the context once tool results passed the cap).
        scoped_messages = _resolve_agent_messages(state, current_agent_name)
        last_messages = list(scoped_messages)

        logger.info(f"📨 Messages in conversation: {len(last_messages)}")

        logger.info("=" * 80)
        if last_messages:
            content_preview = sanitize_history(last_messages)
            content_preview = json.dumps(content_preview[-5:], indent=2)
            logger.info(f"📝 Content preview: {content_preview}")

        logger.info("=" * 80)

        instructions, profile, recalled, recall_updates = await _memory_for_turn(state)

        try:
            final_prompt = system_prompt.replace("{current_time}", current_time)
            messages = (
                [SystemMessage(content=final_prompt)]
                + _memory_messages(instructions, profile, recalled)
                + _summary_messages(state)
                + last_messages
            )
            messages = clean_unmatched_tool_calls(messages)
            logger.info(
                f"🤖 Sending messages to LLM with {count_tokens(messages)} tokens"
            )
            msg = await llm_with_tools.ainvoke(messages)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.error("🚫 Rate limit hit - stopping execution")
                return {
                    "messages": [
                        AIMessage(
                            content="[ERROR] Rate limit reached. Please retry later.",
                            name=current_agent_name,
                            additional_kwargs={"timestamp": current_time},
                        )
                    ]
                }
            logger.error(f"HTTP error: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"🚫 Network error - no internet connection: {e}")
            return {
                "messages": [
                    AIMessage(
                        content="[ERROR] Network unavailable. Check connection.",
                        name=current_agent_name,
                        additional_kwargs={"timestamp": current_time},
                    )
                ]
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

        raw_content = msg.content if msg.content else ""
        final_content = raw_content

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for i, tool_call in enumerate(msg.tool_calls, 1):
                logger.info(f"   Tool #{i}:")
                logger.info(f"      Name: {tool_call.get('name', 'N/A')}")
                logger.info(
                    f"      Args: {json.dumps(tool_call.get('args', {}), indent=10)}"
                )
                # logger.info(f"      ID: {tool_call.get('id', 'N/A')}")

        if hasattr(msg, "content") and msg.content and not msg.tool_calls:
            content_preview = (
                msg.content[:1000] + "..."
                if len(msg.content) > 1000000
                else msg.content
            )

        logger.info("=" * 80)

        agent_message = AIMessage(
            content=final_content,
            name=current_agent_name,
            tool_calls=getattr(msg, "tool_calls", []),
            additional_kwargs={"timestamp": current_time},
        )
        try:
            has_tools = bool(getattr(msg, "tool_calls", []))
            if final_content and not has_tools:
                await log_event(
                    thread_id=thread_id,
                    actor=current_agent_name,
                    message=final_content,
                    metadata={
                        "request_num": request_num,
                        "type": "content",
                    },
                )

            if has_tools:
                await log_event(
                    thread_id=thread_id,
                    actor=current_agent_name,
                    message=f"{', '.join([format_tool_to_text(tc.get('name', ''), json.dumps(tc.get('args', {}))) for tc in msg.tool_calls])}",
                    metadata={"type": "tool_call"},
                )

            # What the worker found and reports back. Tool-call rows are left out of
            # long-term memory, so the result is logged on its own for memory to learn from.
            for tc in getattr(msg, "tool_calls", None) or []:
                handoff_result = (tc.get("args") or {}).get("message")
                if tc.get("name") == "work_completion" and handoff_result:
                    await log_event(
                        thread_id=thread_id,
                        actor=current_agent_name,
                        message=handoff_result,
                        metadata={"request_num": request_num, "type": "handoff_result"},
                    )
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")

        message_key = AGENT_MESSAGE_KEY.get(current_agent_name, "messages")
        return {
            "messages": [agent_message],
            message_key: [agent_message],
            "current_agent": current_agent_name,
            **recall_updates,
        }

    return agent_node


def code_execution_factory(llm, tool_sets, agent_name: str):
    """Create the code execution node that runs CodeExecutionAgent workflows with user permission checks and local Port 9000 sandbox."""

    async def code_executor(state: State, config: RunnableConfig):
        """Execute one code-agent turn and return code or execution output depending on approval state."""
        import re
        current_time = get_current_time()
        current_agent_name = agent_name
        thread_id = _thread_id(config)

        scoped_messages = _resolve_agent_messages(state, current_agent_name)
        
        # Check if the user is approving a previously generated code block
        is_approved = False
        code_to_run = None
        
        last_msg = scoped_messages[-1] if scoped_messages else None
        if last_msg and last_msg.type == "human":
            content_lower = (last_msg.content or "").strip().lower()
            if content_lower in ["approve", "yes", "run", "ok", "run it"]:
                # Traverse backward to find the last AI message with python code
                for msg in reversed(scoped_messages):
                    if msg.type == "ai" and "```python" in msg.content:
                        match = re.search(r"```python\n(.*?)\n```", msg.content, re.DOTALL)
                        if match:
                            code_to_run = match.group(1)
                            is_approved = True
                            break

        if is_approved and code_to_run:
            logger.info("Code execution approved by user! Running...")
            try:
                agent = CodeExecutionAgent(llm, tool_sets)
                # Parse intent from prior messages (exclude the latest 'approve' human message)
                task_spec = await agent._resolve_intent(scoped_messages[:-1])
                tool_map = agent._create_tool_map(task_spec.required_tools_hint)
                
                # Execute in the sacrificial Python sandbox server
                msg = await agent._execute_in_sandbox(code_to_run, tool_map)
                
                if msg.get("status") == "success":
                    summary = msg.get("summary", "Code executed successfully.")
                    full_output = json.dumps(msg.get("full_output", {}), indent=2)
                    
                    await log_event(
                        thread_id=thread_id,
                        actor="code_agent",
                        message=f"PLAN: {task_spec.primary_goal}\n\nEXECUTED CODE:\n```python\n{code_to_run}\n```",
                        metadata={"type": "code_execution"}
                    )
                    await log_event(
                        thread_id=thread_id,
                        actor="code_agent",
                        message=f"EXECUTION RESULT:\nStatus: success\nSummary: {summary}\nDetails: {full_output}",
                        metadata={"type": "code_output", "status": "success"}
                    )
                    
                    response_text = f"Code execution completed successfully.\n\n**Summary**:\n{summary}\n\n**Output details**:\n```json\n{full_output}\n```"
                else:
                    error_msg = msg.get("error", "Unknown sandbox error")
                    response_text = f"Code execution failed with error:\n\n```\n{error_msg}\n```"
                    
                agent_message = AIMessage(
                    content=response_text,
                    name=current_agent_name,
                    additional_kwargs={"timestamp": current_time}
                )
                
                return {
                    "messages": [agent_message],
                    "code_messages": [agent_message],
                    "current_agent": "supervisor" # Handoff back to supervisor
                }
            except Exception as e:
                logger.error(f"Error running approved code: {e}")
                err_msg = AIMessage(content=f"Error executing code: {str(e)}", name=current_agent_name)
                return {
                    "messages": [err_msg],
                    "code_messages": [err_msg],
                    "current_agent": "supervisor"
                }
        else:
            # Generate the Python code first and request user approval
            logger.info("Generating code and requesting user approval...")
            try:
                agent = CodeExecutionAgent(llm, tool_sets)
                last_messages = trim_messages(
                    scoped_messages,
                    max_tokens=30000,
                    strategy="last",
                    token_counter=count_tokens,
                    include_system=True,
                    start_on="human",
                )
                task_spec = await agent._resolve_intent(last_messages)
                
                if not task_spec.required_tools_hint and not task_spec.primary_goal:
                    err_msg = AIMessage(content="Could not parse intent for code execution.", name=current_agent_name)
                    return {
                        "messages": [err_msg],
                        "code_messages": [err_msg],
                        "current_agent": "supervisor"
                    }
                    
                tool_schemas = await agent._load_tool_schemas(task_spec.required_tools_hint)
                code_prompt = agent._build_code_generation_prompt(task_spec, tool_schemas)
                generated_code = await agent._generate_code(code_prompt)
                
                approval_request = (
                    f"I have generated the following Python code to address your request:\n\n"
                    f"```python\n{generated_code}\n```\n\n"
                    f"**Please review the code and reply with 'approve' to execute it.**"
                )
                
                agent_message = AIMessage(
                    content=approval_request,
                    name=current_agent_name,
                    additional_kwargs={"timestamp": current_time}
                )
                
                return {
                    "messages": [agent_message],
                    "code_messages": [agent_message],
                    "current_agent": current_agent_name # Retain context so the approval is routed directly back
                }
            except Exception as e:
                logger.error(f"Error generating code: {e}")
                err_msg = AIMessage(content=f"Error generating code: {str(e)}", name=current_agent_name)
                return {
                    "messages": [err_msg],
                    "code_messages": [err_msg],
                    "current_agent": "supervisor"
                }

    return code_executor



def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " ...(cut)"


def _transcript(messages: list) -> str:
    """Readable lines for the summarizer: who said what, which tools were called and
    what they returned (long text cut), instead of the raw message objects."""
    lines = []
    for m in messages:
        content = m.content if isinstance(m.content, str) else json.dumps(m.content, default=str)
        if isinstance(m, HumanMessage):
            lines.append(f"{m.name or 'User'}: {_clip(content, 1500)}")
        elif isinstance(m, AIMessage):
            speaker = m.name or "assistant"
            if content:
                lines.append(f"{speaker}: {_clip(content, 1500)}")
            for tc in m.tool_calls or []:
                args = json.dumps(tc.get("args", {}), default=str)
                lines.append(f"{speaker} called {tc.get('name')}({_clip(args, 800)})")
        elif isinstance(m, ToolMessage):
            lines.append(f"  -> result: {_clip(content, 400)}")
    return "\n".join(lines)


async def summerizer_node(state: State, config: RunnableConfig):
    """Condense the older turns of the thread into the running summary and remove them.

    Each message list over the limit (the shared one and each agent's own) is cut at a
    turn boundary (core.state.plan_summary), so tool calls stay with their results. The
    summary is shown to the supervisor and the workers. If the model call fails nothing
    is removed and the next turn tries again.
    """
    logger.info("📝 Summarizer node activated to condense conversation history.")
    plan = plan_summary(state)
    if not plan:
        return {}

    # One transcript of what is being archived: the shared list in order, then what
    # lives only in an agent's own list (handoffs to workers and their results).
    shared = plan.get("messages", [])
    seen_ids = {m.id for m in shared}
    agent_only = []
    for channel, archived in plan.items():
        if channel == "messages":
            continue
        for m in archived:
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                agent_only.append(m)

    transcript = _transcript(shared)
    if agent_only:
        transcript += "\n\nHandoffs between agents in these turns:\n" + _transcript(agent_only)

    prompt_content = (
        f"CURRENT SUMMARY:\n{state.get('summary') or '(none yet)'}\n\n"
        f"NEW CHAT MESSAGES:\n{transcript}"
    )
    try:
        llm = build_llm()
        cleaned = await llm.ainvoke(
            [
                SystemMessage(content=prompt_texts.HISTORY_SUMMARIZE_PROMPT),
                SystemMessage(content=prompt_content),
            ]
        )
        summarized_content = cleaned.content
    except Exception as e:
        logger.error(f"Summarizing failed; nothing removed, will retry next turn: {e}")
        return {}
    if not summarized_content:
        logger.error("Summarizer returned nothing; nothing removed, will retry next turn.")
        return {}

    updates = {"summary": summarized_content}
    archived_count = 0
    for channel, archived in plan.items():
        removals = [RemoveMessage(id=m.id) for m in archived if m.id]
        if removals:
            updates[channel] = removals
            archived_count += len(removals)

    logger.info(
        "📊 Archived "
        + ", ".join(f"{len(archived)} from {channel}" for channel, archived in plan.items())
    )

    try:
        await log_event(
            thread_id=_thread_id(config),
            actor="summerizer_node",
            message=f"summerized content: {summarized_content}",
            metadata={"archived_messages": archived_count},
        )
    except Exception as e:
        logger.error(f"Failed to log summarizer audit event: {e}")

    return updates


# One memory run at a time, across threads: saves run on the background saver's thread
# (with its own event loop), and two runs must not store the same logs twice.
_flush_lock = threading.Lock()

# Token budget for one knowledge-graph extraction call. A long backlog is sent in batches
# (one thread per batch), and each stored batch moves the progress mark.
KG_BATCH_TOKENS = 6000


async def flush_memory(reason: str = "", logged_before: datetime | None = None) -> dict:
    """Store every log not yet in long-term memory, from all threads.

    Runs the knowledge-graph and episodic-RAG pipelines over the human_logs rows past each
    pipeline's progress mark (see utils.memory_manager). A pipeline moves its mark only
    after a batch is stored, so a failed run is retried next time instead of being skipped.

    logged_before: only store logs written before this time.
    Returns {pipeline: True if it is now up to date}.
    """
    with _flush_lock:
        logger.info(f"🧠 Saving long-term memory ({reason or 'requested'})")
        return {
            "knowledge_graph": await updation_knowledge_graph(MEMORY_DB, logged_before),
            "episodic_rag": await updation_episodic_rag(MEMORY_DB, logged_before),
        }


class BackgroundMemorySaver:
    """Saves long-term memory on a background thread, so a chat is not held up while
    the knowledge-graph model calls and embeddings run.

    request() starts a save. If one is already running, one more run is queued, so logs
    written in the meantime are included. wait() blocks until saving is done (used on
    exit). Each save runs flush_memory in its own event loop on the saver's thread. The
    thread is a daemon: a forced quit (Ctrl+C) does not wait for it, and whatever it had
    not stored yet is stored at the next startup.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._queued = None  # (reason, logged_before) of the next run
        self.last_result = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def request(self, reason: str, logged_before: datetime | None = None):
        with self._lock:
            if self.running:
                # Keep the widest queued run: no time limit covers everything.
                queued_before = self._queued[1] if self._queued else None
                if (
                    self._queued is None
                    or logged_before is None
                    or (queued_before is not None and logged_before > queued_before)
                ):
                    self._queued = (reason, logged_before)
                return
            self._thread = threading.Thread(
                target=self._run,
                args=(reason, logged_before),
                name="memory-saver",
                daemon=True,
            )
            self._thread.start()

    def _run(self, reason: str, logged_before: datetime | None):
        while True:
            try:
                self.last_result = asyncio.run(flush_memory(reason, logged_before))
            except Exception as e:
                logger.error(f"Background memory save failed: {e}")
                self.last_result = {pipeline: False for pipeline in MEMORY_PIPELINES}
            if not all(self.last_result.values()):
                failed = ", ".join(p for p, ok in self.last_result.items() if not ok)
                logger.warning(
                    f"Memory not fully saved ({failed}); it will be retried next time."
                )
            with self._lock:
                if self._queued is None:
                    self._thread = None
                    return
                reason, logged_before = self._queued
                self._queued = None

    def wait(self, timeout: float | None = None) -> bool:
        """Block until saving is done. Returns False if it is still running at timeout."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return not self.running


# The process-wide saver used by main.py and the new-day memory node.
memory_saver = BackgroundMemorySaver()


async def _last_log_id(db, logged_before: datetime | None = None) -> int:
    """Highest human_logs id (written before logged_before, if given)."""
    query = "SELECT COALESCE(MAX(id), 0) FROM human_logs"
    params = ()
    if logged_before is not None:
        query += " WHERE timestamp < ?"
        params = (logged_before.isoformat(),)
    async with db.execute(query, params) as cursor:
        return (await cursor.fetchone())[0]


def memory_node_factory():
    """Create the memory maintenance node.

    route_start sends a turn here when logs from an earlier day are not in long-term
    memory yet (a session that ran past midnight). The node hands those logs to the
    background saver and the turn carries on; today's logs are stored on /new, /thread
    or exit. It also refreshes the memory timestamps in graph state.
    """

    async def memory_node(state: State, config: RunnableConfig):
        """Start saving earlier days' logs and return state update fields."""
        start_of_today = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        memory_saver.request("new day", logged_before=start_of_today)

        now_float = time.time()
        return {
            "last_knowledgegraph_timestamp": now_float,
            "last_memory_timestamp": now_float,
        }

    return memory_node


async def updation_episodic_rag(db_path=MEMORY_DB, logged_before=None) -> bool:
    """Index the logs the episodic RAG has not stored yet. Returns True when up to date."""
    try:
        async with aiosqlite.connect(db_path) as db:
            after_id = await get_memory_progress(db, "episodic_rag")
            up_to_id = await _last_log_id(db, logged_before)

        if up_to_id <= after_id:
            logger.info("Episodic RAG is up to date.")
            return True

        logger.info(f"🔄 Episodic RAG: indexing logs {after_id + 1}-{up_to_id}.")
        rag = EpisodicRAG(db_path=db_path)
        chunks = await rag.custom_text_splitters(log_id_range=(after_id, up_to_id))

        if chunks and not rag.index_creation(chunks):
            logger.error(
                "Episodic RAG indexing failed; these logs will be retried next time."
            )
            return False

        async with aiosqlite.connect(db_path) as db:
            await set_memory_progress(db, "episodic_rag", up_to_id)
        logger.info(f"✅ Episodic RAG stored {len(chunks)} chunks.")
        return True
    except Exception as e:
        logger.error(
            f"Episodic RAG update failed; these logs will be retried next time: {e}"
        )
        return False


# Whose log lines the knowledge graph learns from, and how each is labelled for the
# extraction model: the user, the supervisor's replies, and what the workers found in
# SIR's apps (their replies and work_completion results).
KG_ACTOR_LABELS = {
    "Human_node": "Yadeesh (user)",
    "supervisor": "Assistant",
    "communication_agent": "Gmail agent (found in Yadeesh's mailbox)",
    "planning_agent": "Calendar agent (found in Yadeesh's calendar and tasks)",
    "document_agent": "Docs agent (found in Yadeesh's Drive and Docs)",
    "data_agent": "Sheets agent (found in Yadeesh's Sheets and Forms)",
    "presentation_agent": "Slides agent (found in Yadeesh's Slides)",
}


def _kg_batches(rows, max_tokens=KG_BATCH_TOKENS):
    """Split (id, thread_id, timestamp, actor, message) rows into one-thread batches
    under max_tokens."""
    batch, batch_tokens = [], 0
    for row in rows:
        tokens = count_tokens(row[4] or "")
        if batch and (row[1] != batch[-1][1] or batch_tokens + tokens > max_tokens):
            yield batch
            batch, batch_tokens = [], 0
        batch.append(row)
        batch_tokens += tokens
    if batch:
        yield batch


async def updation_knowledge_graph(db_path: str = MEMORY_DB, logged_before=None) -> bool:
    """Extract facts from the logs the knowledge graph has not stored yet and apply them.

    Returns True when up to date. Stops at the first batch that fails (e.g. the model is
    unreachable) so it is retried next time; batches stored before it stay stored.
    """
    try:
        # Reuse the shared singleton (app_tools.tools.rag_tools) instead of
        # opening a second kuzu.Database on the same path — kuzu does not
        # support multiple concurrent connections to one database file from
        # the same process.
        from app_tools.tools.rag_tools import get_kg_instance, invalidate_profile

        async with aiosqlite.connect(db_path) as db:
            after_id = await get_memory_progress(db, "knowledge_graph")
            up_to_id = await _last_log_id(db, logged_before)
            if up_to_id <= after_id:
                logger.info("Knowledge graph is up to date.")
                return True

            # Only the conversation itself: the user's messages, and the replies and
            # results the agents give. Tool calls, tool output, code-agent output and
            # summaries never reach the graph, whatever is logged in the future.
            target_actors = tuple(KG_ACTOR_LABELS)
            query = f"""
                SELECT id, thread_id, timestamp, actor, message
                FROM human_logs
                WHERE id > ? AND id <= ?
                AND actor IN ({", ".join("?" for _ in target_actors)})
                AND (
                    actor = 'Human_node'
                    OR json_extract(metadata, '$.type') IN ('content', 'handoff_result')
                )
                ORDER BY id ASC;
            """
            async with db.execute(
                query, (after_id, up_to_id, *target_actors)
            ) as cursor:
                rows = await cursor.fetchall()

        logger.info(
            f"🔄 Knowledge graph: {len(rows)} new log entries (logs {after_id + 1}-{up_to_id})."
        )

        kg = get_kg_instance() if rows else None
        for batch in _kg_batches(rows):
            extraction_context = "\n".join(
                f"{KG_ACTOR_LABELS.get(actor, actor)}: {msg}"
                for _id, _thread, _ts, actor, msg in batch
            )
            _id, thread, timestamp, _actor, _msg = batch[0]
            learned_from = f"conversation (thread {thread}) on {str(timestamp)[:10]}"
            store_facts(kg, extraction_context, learned_from)
            async with aiosqlite.connect(db_path) as db:
                await set_memory_progress(db, "knowledge_graph", batch[-1][0])

        # Also covers the rows the query filters out (tool calls, code agent output).
        async with aiosqlite.connect(db_path) as db:
            await set_memory_progress(db, "knowledge_graph", up_to_id)

        if kg is not None:
            invalidate_profile()
            kg.visualize()
        logger.info("✅ Knowledge graph update process completed successfully.")
        return True

    except Exception as e:
        logger.error(
            f"Knowledge graph update failed; the remaining logs will be retried next time: {e}"
        )
        return False


def _keywords(value) -> str:
    """search_keywords as stored: comma-separated (the model returns a list or a string)."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value or "")


def store_facts(kg, text: str, learned_from: str = "", explicit: bool = False) -> dict:
    """Extract entities and relationships from text and write them to the knowledge graph.

    Used for conversation logs (memory saves) and for what the user explicitly asks to
    store (add_information_to_knowledge_graph, explicit=True), with the same extraction
    and validation rules either way; types and relationship names are normalized.

    Raises if the model cannot be reached, so a memory save retries the batch later. A
    reply that is not valid JSON is retried once; if it fails again the text is skipped
    (and logged), so one bad batch cannot block every later log.

    Returns {"status": "stored" | "nothing_new" | "skipped", "entities": n, "relationships": n}.
    """
    from rag.knowledge_graph import normalize_candidates

    counts = {"entities": 0, "relationships": 0}
    prompt = prompt_texts.KNOWLEDGE_GRAPH_EXTRACTION_PROMPT
    if explicit:
        prompt += "\n\n" + prompt_texts.KNOWLEDGE_GRAPH_EXPLICIT_NOTE

    candidates_json = kg.generate_entity_relation(text, prompt)
    if candidates_json is None:
        candidates_json = kg.generate_entity_relation(text, prompt)
    if candidates_json is None:
        logger.error("Knowledge graph extraction returned invalid JSON twice; skipping this text.")
        return {"status": "skipped", **counts}
    normalize_candidates(candidates_json)

    logger.info(
        f"Extracted candidates for KG update: {json.dumps(candidates_json, indent=2)}"
    )

    candidates = candidates_json.get("candidates") or {}
    if not candidates.get("entities") and not candidates.get("relationships"):
        logger.info("🔍 No valid entities or relationships found in this text.")
        return {"status": "nothing_new", **counts}
    entities = candidates.get("entities") or []

    types_df = kg.search_similar_node(entities)

    validation_prompt = prompt_texts.KNOWLEDGE_GRAPH_VALIDATION_PROMPT
    final_update_json = kg.validate_entity_relation(types_df, candidates_json, validation_prompt)
    if final_update_json is None:
        final_update_json = kg.validate_entity_relation(types_df, candidates_json, validation_prompt)
    if final_update_json is None:
        logger.error("Knowledge graph validation returned invalid JSON twice; skipping this text.")
        return {"status": "skipped", **counts}
    resolution = final_update_json.get("resolution", {})

    logger.info(
        f"The validated KG update resolution: {json.dumps(resolution, indent=2)}"
    )
    for entity in resolution.get("entities", []):
        action = entity.get("action", "DISCARD").upper()
        if action in ("CREATE", "UPDATE") and entity.get("id"):
            kg.add_entity(
                node_id=entity["id"],
                node_type=entity.get("type") or "Concept",
                search_keywords=_keywords(entity.get("search_keywords")),
                description=entity.get("description") or "",
                learned_from=learned_from,
            )
            counts["entities"] += 1

    for rel in resolution.get("relationships", []):
        action = rel.get("action", "DISCARD").upper()
        if not rel.get("source") or not rel.get("target"):
            continue
        # UPDATE is stored like CREATE: add_relationship refreshes an existing edge and
        # never renames other relationships between the same two entities.
        if action in ("CREATE", "UPDATE"):
            kg.add_relationship(
                source=rel["source"],
                target=rel["target"],
                relation_type=rel.get("relation_type") or "RELATED_TO",
            )
            counts["relationships"] += 1

    status = "stored" if counts["entities"] or counts["relationships"] else "nothing_new"
    return {"status": status, **counts}


# What a worker starts with: the user's own words, never a paraphrase, plus a fixed instruction.
WORKER_HANDOFF_TEMPLATE = (
    "[Handoff from supervisor]\n"
    "User request (the user's exact words):\n{request}\n"
    "{context}"
    "\nDo the part of this request that your tools cover. When you are done, call "
    "`work_completion` with the result; if part of the request needs another app, say what is left."
)

# A look-up for another worker: find information only, so nothing is done out of order
# (e.g. emailing a schedule before the meetings are booked).
WORKER_LOOKUP_TEMPLATE = (
    "[Look-up request from supervisor]\n"
    "User request (the user's exact words, for background only; other agents handle the rest of it):\n{request}\n"
    "{context}"
    "\nFind and return only this: {lookup}\n"
    "Use your tools to look it up. Do not create, change, send or delete anything in this handoff. "
    "When you are done, call `work_completion` with what you found (the actual names, addresses, dates, "
    "IDs or text), or say plainly that it is not in your app."
)

# The code agent has no `work_completion` tool: it generates code, asks for approval and
# hands back to the supervisor itself after running it.
CODE_HANDOFF_TEMPLATE = (
    "[Handoff from supervisor]\n"
    "User request (the user's exact words):\n{request}\n"
    "{context}"
)


def _latest_user_request(state: dict) -> str:
    """The user's latest message, verbatim (user messages are the unnamed HumanMessages)."""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage) and not m.name:
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


@tool
def route_to_agent(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    agent: str,
    context: str = "",
    lookup: str = "",
) -> Command:
    """
    Route the conversation to the correct specialized agent.

    The supervisor's job is route when request is based on this agents domain else they can respond directly— do not attempt to perform complex actions
    yourself. The moment you identify the user's intent, route immediately.

    AGENT DOMAINS:
    - communication_agent: Gmail, sending emails, checking email.
    - planning_agent: Google Calendar, Google Tasks, creating meetings, scheduling.
    - document_agent: Google Drive, Google Docs, document creation, drive lookup.
    - data_agent: Google Sheets, Google Forms, spreadsheets, tables.
    - presentation_agent: Google Slides, presentations.
    - code_agent: Executing Python code, complex computations, data science sandboxing.

    The worker automatically receives the user's latest message word for word.

    context: Optional. Only facts the worker cannot find itself: results from earlier
      workers (names, addresses, dates, IDs, text) or what the user said in earlier turns.
      Leave it empty on a first handoff. Never rephrase the request or add instructions.
    lookup: Optional. Set it only to ask this worker to FIND information another worker
      needs (e.g. "the names and email addresses of Yadeesh's direct reports"). The worker
      then only looks it up and changes nothing. Leave it empty for a normal handoff.
    """
    if agent == "code_agent":
        import json
        from pathlib import Path
        enabled_tools_file = Path(__file__).resolve().parent.parent / "data" / "enabled_tools.json"
        is_enabled = True
        if enabled_tools_file.exists():
            try:
                with open(enabled_tools_file, "r") as f:
                    config = json.load(f)
                    is_enabled = config.get("code_agent", True)
            except Exception:
                pass
                
        if not is_enabled:
            logger.warning("Attempted to route to code_agent but it is disabled by user.")
            return Command(
                update={
                    "supervisor_messages": [
                        ToolMessage(
                            content="Error: code_agent is currently disabled by user settings. Please perform the task using other agents (e.g. data_agent, document_agent) or explain directly.",
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "messages": [
                        ToolMessage(
                            content="Error: code_agent is currently disabled by user settings.",
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "current_agent": "supervisor",
                }
            )
    supervisor_closure = ToolMessage(
        content=f"Successfully routed user to {agent}.",
        tool_call_id=tool_call_id,
    )

    context_block = f"\nContext from earlier steps:\n{context.strip()}\n" if context and context.strip() else ""
    request = _latest_user_request(state)
    if agent == "code_agent":
        seed_text = CODE_HANDOFF_TEMPLATE.format(request=request, context=context_block)
    elif lookup and lookup.strip():
        seed_text = WORKER_LOOKUP_TEMPLATE.format(request=request, context=context_block, lookup=lookup.strip())
    else:
        seed_text = WORKER_HANDOFF_TEMPLATE.format(request=request, context=context_block)
    worker_seed = HumanMessage(content=seed_text, name="supervisor")

    agent_key_map = {
        "communication_agent": "communication_messages",
        "planning_agent": "planning_messages",
        "document_agent": "document_messages",
        "presentation_agent": "presentation_messages",
        "data_agent": "data_messages",
        "code_agent": "code_messages",
    }
    store_msg = agent_key_map.get(agent, "messages")

    return Command(
        update={
            store_msg: [worker_seed],
            "supervisor_messages": [supervisor_closure],
            "messages": [
                ToolMessage(
                    content=f"Successfully routed user to {agent}.",
                    tool_call_id=tool_call_id,
                )
            ],
            "current_agent": agent,
        }
    )


@tool
def work_completion(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    message: str,
) -> Command:
    """
    Call this tool to signal that you have completed your specialized task,
    need to hand off control back to the supervisor, or require delegation.

    message: A summary of what you did, the final result, and any information or questions 
      that the supervisor needs to relay back to the user or use for further planning.
    """
    current_agent = state.get("current_agent", "supervisor")

    agent_key_map = {
        "communication_agent": "communication_messages",
        "planning_agent": "planning_messages",
        "document_agent": "document_messages",
        "presentation_agent": "presentation_messages",
        "data_agent": "data_messages",
        "code_agent": "code_messages",
    }

    store_msg = agent_key_map.get(current_agent, "messages")

    worker_closure = ToolMessage(
        content="Handoff successful. Stop generating and wait for supervisor.",
        tool_call_id=tool_call_id,
    )

    supervisor_seed = HumanMessage(
        content=f"[{current_agent} to supervisor] Handoff. Result: {message}",
        name=current_agent,
    )

    return Command(
        update={
            store_msg: [worker_closure],
            "messages": [
                ToolMessage(
                    content="Handoff successful. Stop generating and wait for supervisor.",
                    tool_call_id=tool_call_id,
                )
            ],
            "supervisor_messages": [supervisor_seed],
            "current_agent": "supervisor",
        }
    )
