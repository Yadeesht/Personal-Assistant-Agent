from datetime import datetime
from typing import Annotated, Optional, List, Dict, Any

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict
from utils.helper import setup_logger, count_tokens
from utils.memory_manager import has_pending_memory

logger = setup_logger(__name__)


def last_value(current: Optional[str], new: Optional[str]) -> Optional[str]:
    """Reducer for current_agent: if two tool calls in one step both set it (e.g. two
    parallel handoffs), keep the last instead of failing the whole graph."""
    return new


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
    # Long-term memory recalled for the user's latest message, looked up once per turn
    # and shared by the supervisor and workers (recalled_for = that message's id).
    recalled_memory: Optional[str]
    recalled_for: Optional[str]
    next: Optional[str]
    current_agent: Annotated[Optional[str], last_value]


# Every message list the summarizer looks after: the shared one and each agent's own.
MESSAGE_CHANNELS = (
    "messages",
    "supervisor_messages",
    "communication_messages",
    "planning_messages",
    "document_messages",
    "presentation_messages",
    "data_messages",
    "code_messages",
)

# Agents a turn can resume in directly: a worker that asked the user something, or the
# code agent waiting for approval of the code it wrote.
RESUMABLE_AGENTS = (
    "communication_agent",
    "planning_agent",
    "document_agent",
    "presentation_agent",
    "data_agent",
    "code_agent",
)

# Short-term memory: a message list over SUMMARY_TRIGGER_TOKENS has its older turns
# condensed into the thread summary, keeping about SUMMARY_KEEP_TOKENS of the newest.
SUMMARY_TRIGGER_TOKENS = 150000
SUMMARY_KEEP_TOKENS = 20000


def plan_summary(state) -> dict:
    """Which old messages to archive, per message list: {channel: [messages]}.

    Only lists over SUMMARY_TRIGGER_TOKENS are condensed. A list is cut only right before
    a HumanMessage (the user's message, or a handoff between agents): no tool call is
    waiting for its result there, so a tool call is never separated from its result.
    Each list keeps its newest turns up to SUMMARY_KEEP_TOKENS, and always the whole
    turn in progress. Empty when nothing needs condensing.
    """
    plan = {}
    for channel in MESSAGE_CHANNELS:
        messages = state.get(channel) or []
        if not messages:
            continue
        tokens = [count_tokens([m]) for m in messages]
        if sum(tokens) <= SUMMARY_TRIGGER_TOKENS:
            continue
        cut_points = [
            i for i, m in enumerate(messages) if i > 0 and isinstance(m, HumanMessage)
        ]
        if not cut_points:
            continue
        keep_from = cut_points[-1]
        for i in reversed(cut_points):
            if sum(tokens[i:]) > SUMMARY_KEEP_TOKENS:
                break
            keep_from = i
        plan[channel] = messages[:keep_from]
    return plan


def route_to_active_agent(state) -> str:
    """The agent that should handle the user's message: the worker or code agent the
    user is talking to, otherwise the supervisor."""
    current_agent = state.get("current_agent")
    if current_agent in RESUMABLE_AGENTS:
        logger.info(f"🔄 Resuming conversation in active agent context: {current_agent}")
        return current_agent
    return "supervisor"


def route_after_memory(state) -> str:
    """After the memory step: condense the thread if it has grown too long, then hand
    the message to the active agent."""
    if plan_summary(state):
        return "summerizer_node"
    return route_to_active_agent(state)


class TaskSpec(BaseModel):
    """Structured task specification for code execution"""

    primary_goal: str
    required_tools_hint: List[str]
    context_variables: Dict[str, Any]
    last_error: Optional[str] = None


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
    # Long-term memory is saved on /new, /thread, exit and at startup (main.py). A session
    # that runs past midnight also saves the earlier day's logs here, at most once a day
    # per thread so a failing pipeline is not retried on every turn.
    start_of_today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    last_memory_ts = state.get("last_memory_timestamp")
    tried_today = (
        isinstance(last_memory_ts, (int, float))
        and datetime.fromtimestamp(last_memory_ts) >= start_of_today
    )

    if not tried_today:
        try:
            if await has_pending_memory(logged_before=start_of_today):
                logger.info("📅 New Day Detected: saving earlier logs to long-term memory.")
                return "memory_update_node"
        except Exception as e:
            logger.error(f"Long-term memory check failed: {e}")

    return route_after_memory(state)
