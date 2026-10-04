import asyncio
import json
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

from config.prompts import HISTORY_SUMMARIZE_PROMPT
from config.settings import DEFAULT_THREAD_ID, MEMORY_DB
from core.codeagent import CodeExecutionAgent
from core.llm import build_llm
from core.state import State
from rag.episodic_rag import EpisodicRAG
from utils.helper import (
    count_tokens,
    format_tool_to_text,
    get_current_time,
    request_counter,
    setup_logger,
)
from utils.memory_manager import log_event, sanitize_history
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
    "presentation_agent": "presentation_messages",
    "data_agent": "data_messages",
    "code_agent": "code_messages",
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

        last_messages = trim_messages(
            scoped_messages,
            max_tokens=30000,
            strategy="last",
            token_counter=count_tokens,
            include_system=True,
            start_on="human",
        )

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

        try:
            has_tools = bool(getattr(response, "tool_calls", []))
            if has_tools:
                await log_event(
                    thread_id=DEFAULT_THREAD_ID,
                    actor=agent_name,
                    message=f"{', '.join([format_tool_to_text(tc.get('name', ''), json.dumps(tc.get('args', {}))) for tc in response.tool_calls])}",
                    metadata={"request_num": request_num, "type": "tool_call"},
                )
            elif response.content:
                await log_event(
                    thread_id=DEFAULT_THREAD_ID,
                    actor=agent_name,
                    message=f"Direct response: {response.content[:500]}",
                    metadata={"request_num": request_num, "type": "content"},
                )
        except Exception as e:
            logger.error(f"Failed to log supervisor event: {e}")

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

        scoped_messages = _resolve_agent_messages(state, current_agent_name)
        last_messages = trim_messages(
            scoped_messages,
            max_tokens=10000,
            strategy="last",
            token_counter=count_tokens,
            include_system=True,
            start_on="human",
        )

        logger.info(f"📨 Messages in conversation: {len(last_messages)}")

        logger.info("=" * 80)
        if last_messages:
            content_preview = sanitize_history(last_messages)
            content_preview = json.dumps(content_preview, indent=2)
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
        try:
            has_tools = bool(getattr(msg, "tool_calls", []))
            if final_content and not has_tools:
                await log_event(
                    thread_id=DEFAULT_THREAD_ID,
                    actor=current_agent_name,
                    message=final_content,
                    metadata={
                        "request_num": request_num,
                        "type": "content",
                    },
                )

            if has_tools:
                await log_event(
                    thread_id=DEFAULT_THREAD_ID,
                    actor=current_agent_name,
                    message=f"{', '.join([format_tool_to_text(tc.get('name', ''), json.dumps(tc.get('args', {}))) for tc in msg.tool_calls])}",
                    metadata={"type": "tool_call"},
                )
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")

        message_key = AGENT_MESSAGE_KEY.get(current_agent_name, "messages")
        return {
            "messages": [agent_message],
            message_key: [agent_message],
            "current_agent": current_agent_name,
        }

    return agent_node


def code_execution_factory(llm, tool_sets, agent_name: str):
    """Create the code execution node that runs CodeExecutionAgent workflows with user permission checks and local Port 9000 sandbox."""

    async def code_executor(state: State):
        """Execute one code-agent turn and return code or execution output depending on approval state."""
        import re
        current_time = get_current_time()
        current_agent_name = agent_name

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
                        thread_id=DEFAULT_THREAD_ID,
                        actor="code_agent",
                        message=f"PLAN: {task_spec.primary_goal}\n\nEXECUTED CODE:\n```python\n{code_to_run}\n```",
                        metadata={"type": "code_execution"}
                    )
                    await log_event(
                        thread_id=DEFAULT_THREAD_ID,
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

    try:
        await log_event(
            thread_id=DEFAULT_THREAD_ID,
            actor="summerizer_node",
            message=f"summerized content: {summarized_content}",
            metadata={
                "archived_messages": len(delete_actions),
                "unremovable_messages": missing_ids_count,
            },
        )
    except Exception as e:
        logger.error(f"Failed to log summarizer audit event: {e}")

    updates = {"summary": summarized_content, "messages": delete_actions}

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


def memory_node_factory():
    """Create the memory maintenance node.

    The returned node updates long-term memory systems (knowledge graph and
    episodic RAG) and refreshes memory-related timestamps in graph state.
    """

    async def memory_node(state: State):
        """Run memory update pipelines and return state update fields."""
        from config.settings import DEFAULT_THREAD_ID, MEMORY_DB
        from core.agent import updation_episodic_rag, updation_knowledge_graph

        now_float = time.time()

        updates = {}

        await updation_knowledge_graph(
            state=state, thread_id=DEFAULT_THREAD_ID, db_path=MEMORY_DB
        )

        await updation_episodic_rag(
            past_summary_date=state.get("last_memory_timestamp", 0.0), db_path=MEMORY_DB
        )

        updates["last_knowledgegraph_timestamp"] = now_float

        updates["last_memory_timestamp"] = now_float
        return updates

    return memory_node


async def updation_episodic_rag(past_summary_date=None, db_path=MEMORY_DB):
    """Update episodic RAG index from memory logs after a given timestamp."""
    try:
        logger.info("🔄 Starting episodic RAG update process.")

        if past_summary_date is None or past_summary_date == 0.0:
            past_summary_date = None
            logger.info("No previous timestamp found, fetching all available logs")

        if past_summary_date is not None:
            if isinstance(past_summary_date, float):
                past_summary_date_iso = datetime.fromtimestamp(
                    past_summary_date
                ).isoformat()
            elif isinstance(past_summary_date, datetime):
                past_summary_date_iso = past_summary_date.isoformat()
            else:
                past_summary_date_iso = str(past_summary_date)

            logger.info(f"Fetching logs after: {past_summary_date_iso}")
        else:
            logger.info("Fetching ALL logs from database")

        rag = EpisodicRAG(db_path=db_path)
        chunks = await rag.custom_text_splitters(past_summary_date=past_summary_date)

        if not chunks:
            logger.info("No chunks generated - no new data to index.")
            return

        rag.index_creation(chunks)
        logger.info("✅ Episodic RAG update process completed successfully.")
    except Exception as e:
        logger.error(f"Episodic RAG update failed: {e}")


async def updation_knowledge_graph(
    state: State, thread_id: str, db_path: str = MEMORY_DB
):
    """Extract new facts from logs and apply create/update ops to knowledge graph."""
    try:
        # Reuse the shared singleton (app_tools.tools.rag_tools) instead of
        # opening a second kuzu.Database on the same path — kuzu does not
        # support multiple concurrent connections to one database file from
        # the same process.
        from app_tools.tools.rag_tools import get_kg_instance

        logger.info("🔄 Starting knowledge graph update process.")
        last_update = state.get("last_knowledgegraph_timestamp", 0.0)

        if isinstance(last_update, float):
            last_update_str = datetime.fromtimestamp(last_update).isoformat()

        elif isinstance(last_update, datetime):
            last_update_str = last_update.isoformat()

        else:
            last_update_str = str(last_update)

        query = """
            SELECT actor, message
            FROM human_logs
            WHERE thread_id = ?
            AND actor IN (?,?)
            AND timestamp > ?
            AND COALESCE(json_extract(metadata, '$.type'), '') != 'tool_call'
            ORDER BY timestamp ASC;
        """

        target_actors = ("Human_node", "supervisor")

        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                query, (thread_id, *target_actors, last_update_str)
            ) as cursor:
                rows = await cursor.fetchall()
                logger.info(f"🔎 Found {len(rows)} new log entries in DB.")

        if not rows:
            logger.info("↩️ No new logs found since last update. Exiting.")  # NEW LOG
            return

        extraction_context = "\n".join([f"{actor}: {msg}" for actor, msg in rows])
        kg = get_kg_instance()
        candidates_json = kg.generate_entity_relation(extraction_context)

        logger.info(
            f"Extracted candidates for KG update: {json.dumps(candidates_json, indent=2)}"
        )

        if not candidates_json.get("candidates", {}).get(
            "entities"
        ) and not candidates_json.get("candidates", {}).get("relationships"):
            logger.info(
                "🔍 No valid entities or relationships found. Exiting update process."
            )
            return
        entities = candidates_json.get("candidates", {}).get("entities", [])

        types_df = kg.search_similar_node(entities)

        final_update_json = kg.validate_entity_relation(types_df, candidates_json)
        resolution = final_update_json.get("resolution", {})

        logger.info(
            f"The validated KG update resolution: {json.dumps(resolution, indent=2)}"
        )
        for entity in resolution.get("entities", []):
            action = entity.get("action", "DISCARD").upper()

            if action == "CREATE":
                kg.add_entity(
                    node_id=entity["id"],
                    node_type=entity.get("type", "unknown"),
                    search_keywords=", ".join(entity.get("search_keywords", [])),
                    description=entity.get("description", ""),
                )
            elif action == "UPDATE":
                kg.add_entity(
                    node_id=entity.get("id"),
                    node_type=entity.get("type") or "unknown",
                    search_keywords=", ".join(entity.get("search_keywords", [])),
                    description=entity.get("description") or "",
                )

        for rel in resolution.get("relationships", []):
            action = rel.get("action", "DISCARD").upper()

            if action == "CREATE":
                kg.add_relationship(
                    source=rel["source"],
                    target=rel["target"],
                    relation_type=rel.get("relation_type", "unknown"),
                )
            elif action == "UPDATE":
                kg.modify_relationship(
                    source=rel["source"],
                    target=rel["target"],
                    relation_type=rel.get("relation_type", "unknown"),
                )
        logger.info("✅ Knowledge graph update process completed successfully.")
        kg.visualize()

    except Exception as e:
        logger.error(f"Knowledge graph update failed: {e}")


@tool
def route_to_agent(
    tool_call_id: Annotated[str, InjectedToolCallId],
    agent: str,
    message: str,
) -> Command:
    """
    Route the conversation to the correct specialized agent.

    The supervisor's job is route when request is based on this agents domain else they can respond directly— do not attempt to perform complex actions
    yourself. The moment you identify the user's intent, route immediately.

    AGENT DOMAINS:
    - communication_agent: Gmail, sending emails, checking email, messaging.
    - planning_agent: Google Calendar, Google Tasks, creating meetings, scheduling.
    - document_agent: Google Drive, Google Docs, document creation, drive lookup.
    - data_agent: Google Sheets, Google Forms, spreadsheets, tables.
    - presentation_agent: Google Slides, presentations.
    - code_agent: Executing Python code, complex computations, data science sandboxing.

    message: The seed context the worker agent will start with. Describe clearly
      what the user is asking and provide any relevant parameters already extracted.
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

    worker_seed = HumanMessage(
        content=f"[Supervisor Handoff]: {message}", name="supervisor"
    )

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
