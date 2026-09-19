import logging
import sqlite3
import tiktoken
from datetime import datetime
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import pytz

import re
import html
import asyncio
import aiosqlite
import json
from config.settings import CHECKPOINT_DB
from config.settings import DEFAULT_THREAD_ID


def clean_email_body(text: str) -> str:
    # 1. Decode HTML entities (e.g., convert &#39; to ')
    text = html.unescape(text)

    # 2. Strip ZWNJ and other invisible junk
    text = re.sub(r"[^\x20-\x7e]", r"", text)

    # 3. Remove repeated special characters (like those divider lines -----)
    text = re.sub(r"[-*=_]{3,}", " ", text)

    # 4. Normalize whitespace
    text = " ".join(text.split())

    text = re.sub(r"\(mailto:[^)]*\)", "", text)

    # Optional: Clean up the [Awesome], [Decent] text left behind
    text = re.sub(r"\[Awesome\]|\[Decent\]|\[Not Great\]", "", text)

    return text


# This class does the short-term memory thing by adding ad summerizing the context in real-time
class AsyncSqliteSaver(AsyncSqliteSaver):
    async def aput(self, config, checkpoint, metadata, new_versions):
        """Save checkpoint with cleaned messages"""

        return await super().aput(config, checkpoint, metadata, new_versions)


mock_tool_sets = {"communication": [], "planning": [], "content": [], "supervisor": []}


async def get_agent_state(thread_id: str):
    async with aiosqlite.connect(str(CHECKPOINT_DB)) as conn:
        checkpointer = AsyncSqliteSaver(conn)

        from core.graph import build_graph

        graph = build_graph(mock_tool_sets, checkpointer)

        config = {"configurable": {"thread_id": thread_id}}

        snapshot = await graph.aget_state(config)

        if not snapshot.values:
            print("❌ No state found for this thread ID.")
            return

        values = snapshot.values
        print("=" * 40)
        print(f"📊 STATE FOR THREAD: {thread_id}")
        print("=" * 40)
        print(
            f"🕒 Last Memory Timestamp: {datetime.fromtimestamp(values.get('last_memory_timestamp')) if values.get('last_memory_timestamp') else 'N/A'}"
        )
        print(f"🧠 Summary: {values.get('summary')}")
        print(f"📨 Total Messages: {len(values.get('messages', []))}")
        print(f"🔜 Next Step: {snapshot.next}")
        print("=" * 40)


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


def setup_logger(name: str = __name__) -> logging.Logger:
    """Configure and return a logger instance"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
    # and "writes" (there is no "messages" table). See
    # utils/memory_manager.py:analyze_checkpoint_db for the same schema.
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


def format_tool_to_text(tool_name, tool_args_str):
    try:
        args = json.loads(tool_args_str)
    except (json.JSONDecodeError, TypeError):
        return f"[Action: {tool_name}] (Args: {tool_args_str})"

    if not isinstance(args, dict):
        return f"[Action: {tool_name}] (Args: {tool_args_str})"

    arg_summary = ", ".join([f"{k}={v}" for k, v in args.items()])
    return f"__Tool Action__: Used {tool_name} with inputs: {arg_summary}"


if __name__ == "__main__":
    asyncio.run(get_agent_state(DEFAULT_THREAD_ID))
