from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from utils.helper import setup_logger, count_tokens

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
    data_messages: Annotated[list, add_messages]
    summary: Optional[str]
    next: Optional[str]
    current_agent: Annotated[Optional[str], last_value]


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
        "data_agent",
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


def internal_agent_route(state: State) -> str:
    """Route from agent node to tools or END"""
    current_agent = state.get("current_agent", "")
    agent_key_map = {
        "communication_agent": "communication_messages",
        "planning_agent": "planning_messages",
        "document_agent": "document_messages",
        "data_agent": "data_messages",
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


def route_start(state: State) -> str:
    messages = state["messages"]

    if count_tokens(messages) > 8000:
        return "summerizer_node"

    # Direct agent resume or fallback to supervisor
    current_agent = state.get("current_agent")
    if current_agent in [
        "communication_agent",
        "planning_agent",
        "document_agent",
        "data_agent",
    ]:
        logger.info(f"🔄 Resuming conversation in active agent context: {current_agent}")
        return current_agent

    return "supervisor"
