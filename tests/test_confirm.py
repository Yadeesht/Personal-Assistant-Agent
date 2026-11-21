"""
Tests for the confirmation gate (core/confirm.py, used by the tool nodes in core/graph.py).

Run with:
    pytest tests/test_confirm.py -v

The gate runs in a real LangGraph graph with a checkpointer: a send_email call pauses
the graph, and the tool runs only when the run is resumed with a yes.
"""

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command  # noqa: E402

from core.confirm import is_approval  # noqa: E402
from core.graph import create_agent_tool_node  # noqa: E402
from core.state import State  # noqa: E402

sent = []


@tool
def send_email(recipient_id: str, subject: str, message: str) -> str:
    """Send an email."""
    sent.append(recipient_id)
    return "sent"


@tool
def search_emails(query: str) -> str:
    """Search emails."""
    return "3 emails"


def _graph():
    builder = StateGraph(State)
    builder.add_node(
        "communication_tools",
        create_agent_tool_node([send_email, search_emails], "communication_messages"),
    )
    builder.add_edge(START, "communication_tools")
    builder.add_edge("communication_tools", END)
    return builder.compile(checkpointer=MemorySaver())


def _call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


EMAIL = {"recipient_id": "yadeesh005@gmail.com", "subject": "Summary", "message": "Hi ..."}


def _pending(graph, config):
    snapshot = asyncio.run(graph.aget_state(config))
    return [i.value for t in snapshot.tasks for i in t.interrupts]


def test_send_email_waits_for_a_yes():
    sent.clear()
    graph, config = _graph(), {"configurable": {"thread_id": "a"}}
    asyncio.run(graph.ainvoke({"communication_messages": [_call("send_email", EMAIL)]}, config))

    assert sent == []  # paused, nothing sent
    (request,) = _pending(graph, config)
    assert request == {"kind": "confirm_tool_calls", "calls": [{"tool": "send_email", "args": EMAIL}]}

    state = asyncio.run(graph.ainvoke(Command(resume={"reply": "Yes, send it!"}), config))
    assert sent == ["yadeesh005@gmail.com"]
    assert state["communication_messages"][-1].content == "sent"
    assert _pending(graph, config) == []


def test_anything_but_yes_cancels_and_reaches_the_agent():
    sent.clear()
    graph, config = _graph(), {"configurable": {"thread_id": "b"}}
    asyncio.run(graph.ainvoke({"communication_messages": [_call("send_email", EMAIL)]}, config))

    reply = "no, that's my own address - use raajan@club.org"
    state = asyncio.run(graph.ainvoke(Command(resume={"reply": reply}), config))
    assert sent == []
    result = state["communication_messages"][-1]
    assert isinstance(result, ToolMessage) and result.tool_call_id == "c1"
    assert "did not approve" in result.content and "raajan@club.org" in result.content
    assert state["messages"][-1].content == result.content  # mirrored to the shared list


def test_other_tools_run_without_asking():
    graph, config = _graph(), {"configurable": {"thread_id": "c"}}
    state = asyncio.run(graph.ainvoke({"communication_messages": [_call("search_emails", {"query": "from:ravi"})]}, config))
    assert state["communication_messages"][-1].content == "3 emails"
    assert _pending(graph, config) == []


def test_only_a_plain_yes_approves():
    for reply in ["yes", "Yes!", "y", "send it", "OK", "go ahead"]:
        assert is_approval(reply), reply
    for reply in ["", "no", "yes but change the subject", "send it to Ravi instead", "wait"]:
        assert not is_approval(reply), reply
