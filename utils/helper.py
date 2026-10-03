import logging
import sqlite3
import sys
import tiktoken
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytz
from rich.console import Console
from rich.logging import RichHandler

import re
import html
import unicodedata
from config.settings import CHECKPOINT_DB


def clean_email_body(text: str) -> str:
    # 1. Decode HTML entities (e.g., convert &#39; to ')
    text = html.unescape(text)

    # 2. Strip ZWNJ and other invisible junk: format/control characters plus the
    #    combining grapheme joiner used as preheader padding. Whitespace stays for
    #    step 4 (so lines don't run together) and visible text such as ₹, — and
    #    accented letters is kept.
    text = "".join(
        ch
        for ch in text
        if ch.isspace()
        or (unicodedata.category(ch) not in ("Cf", "Cc") and ch != "͏")
    )

    # 3. Remove repeated special characters (like those divider lines -----)
    text = re.sub(r"[-*=_]{3,}", " ", text)

    # 4. Normalize whitespace
    text = " ".join(text.split())

    text = re.sub(r"\(mailto:[^)]*\)", "", text)

    # Optional: Clean up the [Awesome], [Decent] text left behind
    text = re.sub(r"\[Awesome\]|\[Decent\]|\[Not Great\]", "", text)

    return text


def count_tokens(messages):
    """
    Count tokens for messages. Handles both:
    - List of message objects with .content attribute
    - List of plain strings
    - Single string
    """
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Handle single string input
    if isinstance(messages, str):
        return len(encoding.encode(messages))

    # Handle list of messages
    num_tokens = 0
    for message in messages:
        if isinstance(message, str):
            # Plain string - just encode it
            num_tokens += len(encoding.encode(message))
        elif hasattr(message, "content"):
            # Message object with content attribute
            num_tokens += 4  # Message formatting overhead
            num_tokens += len(encoding.encode(str(message.content)))
        else:
            # Unknown type, try to convert to string
            num_tokens += len(encoding.encode(str(message)))

    # Add reply priming tokens only for message objects (not plain strings)
    if messages and not isinstance(messages[0], str):
        num_tokens += 2

    return num_tokens


try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

console = Console()


def setup_logger(name: str = __name__) -> logging.Logger:
    """Configure and return a logger instance with clean Rich formatting"""
    root_logger = logging.getLogger()
    if not any(isinstance(h, RichHandler) for h in root_logger.handlers):
        root_logger.handlers = [
            h for h in root_logger.handlers if not isinstance(h, logging.StreamHandler)
        ]
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            markup=False,
            omit_repeated_times=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    return logging.getLogger(name)


class RequestTracker:
    """Tracks LLM calls per agent, per turn, and across the session.
    Detects routing loops and outliers automatically.
    Supports existing `request_counter[key] += 1` syntax unchanged.
    """

    OUTLIER_THRESHOLD = 4  # calls by one agent in a single turn before warning

    def __init__(self):
        self._totals: dict = {}  # cumulative across the whole session
        self._turn_counts: dict = {}  # calls per agent in the current turn
        self._turn_history: list = []  # summary of every completed turn
        self._turn_label: str = ""  # human-readable label for the current turn

    # ── dict-compatible interface ──────────────────────────────────────────────

    def __getitem__(self, key: str) -> int:
        return self._totals.get(key, 0)

    def __setitem__(self, key: str, value: int):
        old = self._totals.get(key, 0)
        self._totals[key] = value
        # detect an increment (e.g. counter[x] += 1) and mirror it to per-turn
        if value == old + 1:
            self._turn_counts[key] = self._turn_counts.get(key, 0) + 1
            turn_count = self._turn_counts[key]
            if turn_count == self.OUTLIER_THRESHOLD:
                _tracker_logger = setup_logger("request_tracker")
                _tracker_logger.warning(
                    f"⚠️  LOOP DETECTED: '{key}' called {turn_count}x in this turn — "
                    f"possible routing loop or repeated failure."
                )

    # ── turn lifecycle ─────────────────────────────────────────────────────────

    def start_turn(self, label: str = ""):
        """Call before each graph.ainvoke to reset per-turn counters."""
        self._turn_counts = {}
        self._turn_label = (
            label[:80] if label else f"turn_{len(self._turn_history) + 1}"
        )

    def end_turn(self) -> dict:
        """Call after each graph.ainvoke. Logs a summary and returns it."""
        outliers = {
            k: v for k, v in self._turn_counts.items() if v >= self.OUTLIER_THRESHOLD
        }
        summary = {
            "turn": self._turn_label,
            "calls": dict(self._turn_counts),
            "total": sum(self._turn_counts.values()),
            "outliers": outliers,
        }
        self._turn_history.append(summary)

        _tracker_logger = setup_logger("request_tracker")
        calls_str = " | ".join(
            f"{k}: {v}" for k, v in sorted(self._turn_counts.items())
        )
        status = f" 🔴 OUTLIERS: {list(outliers.keys())}" if outliers else " ✅"
        _tracker_logger.info(
            f"📊 Turn summary [{summary['total']} calls]{status} — {calls_str or 'none'}"
        )
        return summary

    # ── session helpers ────────────────────────────────────────────────────────

    def session_total(self) -> int:
        return sum(self._totals.values())

    def session_summary(self) -> dict:
        return {
            "total_calls": self.session_total(),
            "by_agent": dict(self._totals),
            "turns": len(self._turn_history),
            "turn_history": self._turn_history,
        }


request_counter = RequestTracker()


def delete_thread_from_db(thread_id: str):
    """Clear memory for a specific thread"""

    # AsyncSqliteSaver stores checkpoints across two tables, "checkpoints"
    # and "writes" (there is no "messages" table).
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    deleted = cursor.rowcount
    cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    conn.commit()
    conn.close()
    print(f"✅ Deleted {deleted} checkpoints for thread: {thread_id}")


def get_current_time():
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    return now.strftime("%Y-%m-%d %H:%M:%S IST")


def sanitize_history(messages):
    clean_history = []

    for msg in messages:
        # 1. Handle Human Messages (includes agent reports sent as HumanMessage)
        if isinstance(msg, HumanMessage):
            entry = {
                "role": "user",
                "name": getattr(msg, "name", None) or "human",
                "content": msg.content[:500] + "..."
                if isinstance(msg.content, str) and len(msg.content) > 500
                else msg.content,
            }
            clean_history.append(entry)

        # 2. Handle AI Messages
        elif isinstance(msg, AIMessage):
            # msg.name is where the actual agent name is stored
            agent_name = getattr(msg, "name", None) or msg.additional_kwargs.get(
                "agent_name", "unknown_agent"
            )
            routed_to = msg.additional_kwargs.get("routed_to")

            entry = {
                "role": "assistant",
                "agent": agent_name,
                "content": msg.content[:500] + "..."
                if isinstance(msg.content, str) and len(msg.content) > 500
                else (msg.content or ""),
            }

            if routed_to:
                entry["routed_to"] = routed_to

            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"name": tool["name"], "args": tool["args"]}
                    for tool in msg.tool_calls
                ]

            clean_history.append(entry)

        # 3. Handle Tool Results
        elif isinstance(msg, ToolMessage):
            result = msg.content
            if isinstance(result, str) and len(result) > 300:
                result = result[:300] + "..."
            clean_history.append(
                {
                    "role": "tool_result",
                    "tool_name": getattr(msg, "name", None) or msg.tool_call_id,
                    "tool_call_id": msg.tool_call_id,
                    "result": result,
                }
            )

    return clean_history
