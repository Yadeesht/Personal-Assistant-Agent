import json

import httpx
from langchain_core.messages import (
    AIMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from config.prompts import HISTORY_SUMMARIZE_PROMPT
from core.llm import build_llm
from core.state import State
from utils.helper import (
    count_tokens,
    get_current_time,
    request_counter,
    sanitize_history,
    setup_logger,
)
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

    return cleaned_messages


AGENT_MESSAGE_KEY = {
    "supervisor": "supervisor_messages",
    "communication_agent": "communication_messages",
    "planning_agent": "planning_messages",
    "document_agent": "document_messages",
    "data_agent": "data_messages",
}


def _resolve_agent_messages(state: State, agent_name: str):
    message_key = AGENT_MESSAGE_KEY.get(agent_name, "messages")
    return state.get(message_key, []) or state.get("messages", [])




def supervisor_node_factory(
    llm_with_tools,
    system_prompt,
    agent_name="supervisor",
):
    """Create the supervisor graph node with isolated supervisor context."""

    async def supervisor_node(state: State):
        request_counter[agent_name] += 1
        request_num = request_counter[agent_name]

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

        try:
            summary = state.get("summary", None)
            if summary:
                summary_msg = SystemMessage(
                    content=f"Conversation Summary of previous messages:\n{summary}"
                )
                last_messages = [summary_msg] + last_messages

            final_prompt = system_prompt.replace("{current_time}", current_time)

            message = [SystemMessage(content=final_prompt)] + last_messages
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

        return {
            "messages": [agent_message],
            "supervisor_messages": [agent_message],
            "current_agent": agent_name,
        }

    return supervisor_node




def agent_node_factory(llm_with_tools, system_prompt, agent_name: str):
    """Create a specialized worker-agent node.

    The returned node invokes the worker LLM with isolated agent context.
    """

    async def agent_node(state: State):

        current_agent_name = agent_name
        request_counter[current_agent_name] += 1
        request_num = request_counter[current_agent_name]

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

        try:
            final_prompt = system_prompt.replace("{current_time}", current_time)
            messages = [SystemMessage(content=final_prompt)] + last_messages
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

        message_key = AGENT_MESSAGE_KEY.get(current_agent_name, "messages")
        update = {
            "messages": [agent_message],
            message_key: [agent_message],
            "current_agent": current_agent_name,
        }
        return update

    return agent_node


async def summerizer_node(state: State):
    """Condense long chat history into a compact summary and prune old messages."""
    logger.info("📝 Summarizer node activated to condense conversation history.")

    messages = state["messages"]

    MAX_RECENT_TOKENS = 4000
    current_tokens = 0
    split_index = 0

    for i in range(len(messages) - 1, -1, -1):
        msg_token_count = count_tokens([messages[i]])

        if (
            current_tokens + msg_token_count > MAX_RECENT_TOKENS
            and (len(messages) - i) > 2
        ):
            split_index = i + 1
            break

        current_tokens += msg_token_count
    if split_index == 0:
        split_index = max(0, len(messages) - 2)

    messages_to_summerize = messages[:split_index]

    logger.info(
        f"📊 Dynamic split: Archiving {len(messages_to_summerize)} messages. Retaining {len(messages) - split_index} messages ({current_tokens} tokens)."
    )

    # right now the prompt is not aware of we sending the summary and to summerize previous messages too
    prompt_content = f"Summary:\n{state.get('summary', '')}\n\n Chat Messages:\n{messages_to_summerize}"
    llm = build_llm()
    messages = [
        SystemMessage(content=HISTORY_SUMMARIZE_PROMPT),
        SystemMessage(content=prompt_content),
    ]
    cleaned = await llm.ainvoke(messages)

    summarized_content = cleaned.content

    delete_actions = []
    missing_ids_count = 0
    for m in messages_to_summerize:
        if m.id:
            delete_actions.append(RemoveMessage(id=m.id))
        else:
            missing_ids_count += 1

    if missing_ids_count > 0:
        logger.warning(
            f"⚠️ Found {missing_ids_count} messages without IDs that cannot be removed."
        )

    updates ={"summary": summarized_content, "messages": delete_actions}

    # The global "messages" channel doesn't mirror every scoped-agent
    # message 1:1 (e.g. supervisor handoff seeds only live in the scoped
    # channel), so per-agent channels never shrink if we only prune here.
    # Mirror the deletion into each scoped channel, but only for ids that
    # actually exist there — add_messages raises if asked to remove an id
    # it doesn't have.
    ids_to_remove = {m.id for m in messages_to_summerize if getattr(m, "id", None)}
    if ids_to_remove:
        for message_key in AGENT_MESSAGE_KEY.values():
            scoped_messages = state.get(message_key, [])
            scoped_ids = {
                m.id for m in scoped_messages if getattr(m, "id", None)
            }
            matched_ids = ids_to_remove & scoped_ids
            if matched_ids:
                updates[message_key] = [
                    RemoveMessage(id=mid) for mid in matched_ids
                ]

    return updates


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
    - document_agent: Google Docs, document creation, document lookup.
    - data_agent: Google Sheets, spreadsheets, tables.

    The worker automatically receives the user's latest message word for word.

    context: Optional. Only facts the worker cannot find itself: results from earlier
      workers (names, addresses, dates, IDs, text) or what the user said in earlier turns.
      Leave it empty on a first handoff. Never rephrase the request or add instructions.
    lookup: Optional. Set it only to ask this worker to FIND information another worker
      needs (e.g. "the names and email addresses of Yadeesh's direct reports"). The worker
      then only looks it up and changes nothing. Leave it empty for a normal handoff.
    """
    supervisor_closure = ToolMessage(
        content=f"Successfully routed user to {agent}.",
        tool_call_id=tool_call_id,
    )

    context_block = f"\nContext from earlier steps:\n{context.strip()}\n" if context and context.strip() else ""
    request = _latest_user_request(state)
    if lookup and lookup.strip():
        seed_text = WORKER_LOOKUP_TEMPLATE.format(request=request, context=context_block, lookup=lookup.strip())
    else:
        seed_text = WORKER_HANDOFF_TEMPLATE.format(request=request, context=context_block)
    worker_seed = HumanMessage(content=seed_text, name="supervisor")

    agent_key_map = {
        "communication_agent": "communication_messages",
        "planning_agent": "planning_messages",
        "document_agent": "document_messages",
        "data_agent": "data_messages",
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
        "data_agent": "data_messages",
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
