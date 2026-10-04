"""Confirmation gate for tool calls that act outside the app and cannot be undone.

Before such a call runs (e.g. send_email), the tool node pauses the graph with
langgraph's interrupt(). main.py shows the user exactly what would be done and resumes
with their answer: the call runs only if they reply yes. Anything else cancels it, and
their words go back to the agent so it can adjust (or ask). The model cannot skip this.
"""

import json

from langchain_core.messages import AIMessage, ToolMessage

from config.settings import CONFIRM_BEFORE_TOOLS

# Replies that approve. Anything else (including "yes, but change ...") cancels.
APPROVAL_REPLIES = {
    "y",
    "yes",
    "yes send",
    "yes send it",
    "send",
    "send it",
    "ok",
    "okay",
    "approve",
    "approved",
    "confirm",
    "go ahead",
    "yep",
    "sure",
}


def is_approval(reply: str) -> bool:
    normalized = "".join(ch for ch in (reply or "").lower() if ch.isalnum() or ch == " ")
    return " ".join(normalized.split()) in APPROVAL_REPLIES


def calls_needing_confirmation(messages: list) -> list:
    """The tool calls in the latest AI message that must be confirmed first."""
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return []
    return [tc for tc in last.tool_calls if tc.get("name") in CONFIRM_BEFORE_TOOLS]


def confirmation_request(calls: list) -> dict:
    """What the user is shown: each call's tool name and arguments."""
    return {
        "kind": "confirm_tool_calls",
        "calls": [{"tool": tc["name"], "args": tc.get("args", {})} for tc in calls],
    }


def declined_messages(messages: list, reply: str) -> list:
    """Tool results for a step the user did not approve: nothing in it is run."""
    last = messages[-1]
    results = []
    for tc in last.tool_calls:
        if tc.get("name") in CONFIRM_BEFORE_TOOLS:
            content = (
                f"Not done: SIR did not approve this {tc['name']} call. SIR said: "
                f"{json.dumps(reply)}. Do not repeat it unchanged: adjust it to what SIR "
                "said (it will be shown to SIR again), or ask SIR what to do."
            )
        else:
            content = (
                "Not run: SIR declined another action in the same step. "
                "Call this again if it is still needed."
            )
        results.append(ToolMessage(content=content, tool_call_id=tc["id"], name=tc["name"]))
    return results
