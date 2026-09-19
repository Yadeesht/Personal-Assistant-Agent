import json
import re
import time
from datetime import datetime
from typing import Annotated, Literal, Optional, List, Dict, Any

import aiosqlite
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from utils.helper import setup_logger, count_tokens

from config.settings import MEMORY_DB

logger = setup_logger(__name__)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    supervisor_messages: Annotated[list, add_messages]
    communication_messages: Annotated[list, add_messages]
    planning_messages: Annotated[list, add_messages]
    document_messages: Annotated[list, add_messages]
    presentation_messages: Annotated[list, add_messages]
    data_messages: Annotated[list, add_messages]
    code_messages: Annotated[list, add_messages]
    summary: Optional[str]
    last_memory_timestamp: Optional[float]
    last_knowledgegraph_timestamp: Optional[float]
    next: Optional[str]
    current_agent: Optional[str]


class TaskSpec(BaseModel):
    """Structured task specification for code execution"""

    primary_goal: str
    required_tools_hint: List[str]
    context_variables: Dict[str, Any]
    last_error: Optional[str] = None


class Route(BaseModel):
    """Routing decision for the supervisor"""

    step: Literal[
        "communication_agent",
        "planning_agent",
        "document_agent",
        "presentation_agent",
        "data_agent",
        "code_agent",
        "FINISH",
    ] = Field(
        description="The next agent to route to, or FINISH if done no other response other than these is allowed"
    )


def _parse_next_from_json(content: str) -> Optional[str]:
    if not content or not isinstance(content, str):
        return None

    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if not json_match:
        return None

    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None

    next_agent = parsed.get("next")
    if not isinstance(next_agent, str):
        return None
    return next_agent


def route_after_supervisor(state: State):
    supervisor_messages = state.get("supervisor_messages", [])
    if not supervisor_messages:
        return "FINISH"

    last_message = supervisor_messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "supervisor_tools"

    return "FINISH"


def route_after_supervisor_tools(state: State):
    current = state.get("current_agent", "supervisor")
    if current in [
        "communication_agent",
        "planning_agent",
        "document_agent",
        "presentation_agent",
        "data_agent",
        "code_agent",
    ]:
        return current
    return "supervisor"


def route_after_communication_tools(state: State):
    return "supervisor" if state.get("current_agent") == "supervisor" else "communication_agent"


def route_after_planning_tools(state: State):
    return "supervisor" if state.get("current_agent") == "supervisor" else "planning_agent"


def route_after_document_tools(state: State):
    return "supervisor" if state.get("current_agent") == "supervisor" else "document_agent"


def route_after_data_tools(state: State):
    return "supervisor" if state.get("current_agent") == "supervisor" else "data_agent"


def route_after_presentation_tools(state: State):
    return "supervisor" if state.get("current_agent") == "supervisor" else "presentation_agent"


def route_after_code_agent(state: State) -> str:
    """Route after the code_agent node runs.

    code_executor sets current_agent back to "code_agent" while it is waiting
    on the user to approve generated code — in that case the turn must end
    here (not fall through to supervisor) so the approval request actually
    reaches the user and the next turn resumes directly in code_agent.
    Once execution has actually happened (success or error), current_agent is
    set to "supervisor" and control hands off normally.
    """
    return "supervisor" if state.get("current_agent") == "supervisor" else "END"


def internal_agent_route(state: State) -> str:
    """Route from agent node to tools or END"""
    current_agent = state.get("current_agent", "")
    agent_key_map = {
        "communication_agent": "communication_messages",
        "planning_agent": "planning_messages",
        "document_agent": "document_messages",
        "presentation_agent": "presentation_messages",
        "data_agent": "data_messages",
        "code_agent": "code_messages",
    }

    message_key = agent_key_map.get(current_agent)
    messages = state.get(message_key, []) if message_key else []
    if not messages:
        messages = state.get("messages", [])
    if not messages:
        return "END"

    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"🔧 Agent requesting {len(last_message.tool_calls)} tool(s)")
        return "tools"

    return "END"


async def route_start(state: State) -> str:

    last_memory_ts = state.get("last_memory_timestamp")

    if isinstance(last_memory_ts, float):
        query_ts = datetime.fromtimestamp(last_memory_ts).isoformat()

    elif isinstance(last_memory_ts, datetime):
        query_ts = last_memory_ts.isoformat()

    else:
        query_ts = str(last_memory_ts)

    should_update_memory = False

    if last_memory_ts is None:
        logger.info("🆕 New Thread Detected: Initializing memory timestamp.")
        should_update_memory = True
    else:
        now_float = time.time()
        SECONDS_IN_DAY = 86400
        IST_OFFSET = 19800

        current_day_ist = int((now_float + IST_OFFSET) // SECONDS_IN_DAY)
        memory_day_ist = int((last_memory_ts + IST_OFFSET) // SECONDS_IN_DAY)

        if current_day_ist > memory_day_ist:
            logger.info("📅 New Day Detected: Triggering memory optimization.")
            query = """
                SELECT timestamp, actor, message, metadata
                FROM human_logs 
                WHERE timestamp > ? 
                AND actor NOT IN ('supervisor_routing', 'summerizer_node')
                AND COALESCE(json_extract(metadata, '$.type'), '') != 'tool_call'
                ORDER BY timestamp ASC
            """
            async with aiosqlite.connect(MEMORY_DB) as db:
                async with db.execute(query, (query_ts,)) as cursor:
                    rows = await cursor.fetchall()
                    logger.info(f"Processing {len(rows)} raw logs...")
            if len(rows) > 0:
                should_update_memory = True

    if should_update_memory:
        return "memory_update_node"

    messages = state["messages"]

    if count_tokens(messages) > 8000:
        return "summerizer_node"

    # Direct agent resume or fallback to supervisor
    current_agent = state.get("current_agent")
    if current_agent in [
        "communication_agent",
        "planning_agent",
        "document_agent",
        "presentation_agent",
        "data_agent",
        "code_agent",
    ]:
        logger.info(f"🔄 Resuming conversation in active agent context: {current_agent}")
        return current_agent

    return "supervisor"
