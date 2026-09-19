from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import aiosqlite
from datetime import datetime
import sqlite3

import msgpack
import pickle
from msgpack import ExtType
import json
import re

from config.settings import MEMORY_DB, CHECKPOINT_DB
from utils.helper import count_tokens, setup_logger

logger = setup_logger(__name__)


async def log_event(thread_id: str, actor: str, message: str, metadata: dict = None):
    """Saves a human-readable log entry to a separate table."""
    async with aiosqlite.connect(MEMORY_DB) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS human_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                timestamp TEXT,
                actor TEXT,
                message TEXT,
                metadata TEXT
            )
        """)
        await db.execute(
            "INSERT INTO human_logs (thread_id, timestamp, actor, message, metadata) VALUES (?, ?, ?, ?, ?)",
            (
                thread_id,
                datetime.now().isoformat(),
                actor,
                message,
                str(metadata or {}),
            ),
        )
        await db.commit()


def analyze_human_logs(
    db_path=None,
    output_file="utils/log_details.txt",
):
    if db_path is None:
        db_path = str(MEMORY_DB)
    try:
        # 1. Connect to your local SQLite file
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 2. Query your new human-readable table
        # We select the columns defined in your log_event function
        query = """
            SELECT thread_id, timestamp, actor, message, metadata 
            FROM human_logs 
            ORDER BY timestamp ASC
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            message = "🕒 No human-readable logs found yet. Start a conversation first!"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(message + "\n")
            conn.close()
            return
        messages = []
        # 3. Open file for writing the clear-text report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("📊 AGENTIC AI - HUMAN READABLE AUDIT LOG\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")

            for thread_id, timestamp, actor, message, metadata in rows:
                f.write(
                    f"[{timestamp}] | THREAD: {thread_id[:8]}... | ACTOR: {actor.upper()}\n"
                )
                f.write(f"MESSAGE: {message}\n")
                messages.append(message)

                # Metadata is stored as a stringified dict in your log_event code
                if metadata and metadata != "{}":
                    f.write(f"METADATA: {metadata}\n")

                f.write("-" * 100 + "\n")

            # 4. Summary Stats
            unique_threads = len(set(row[0] for row in rows))
            summary = "\nSUMMARY:\n"
            summary += f"Total Events Logged: {len(rows)}\n"
            summary += f"Active Threads: {unique_threads}\n"
            summary += f"Total token count in messages: {sum(count_tokens(msg) for msg in messages)} tokens\n"
            f.write(summary)

        print(f"✅ Audit report successfully written to {output_file}")
        conn.close()

    except sqlite3.OperationalError:
        logger.error(
            "❌ Table 'human_logs' does not exist yet. Ensure an event has been logged first."
        )
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {e}")


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


OUTPUT_FILE = "utils/checkpoint_text_dump.txt"


def ext_hook(code, data):
    if code == 5:  # LangChain message wrapper
        try:
            return pickle.loads(data)
        except Exception:
            return {"__raw_message__": data}
    return ExtType(code, data)


def decode_checkpoint(blob):
    return msgpack.unpackb(
        blob,
        raw=False,
        ext_hook=ext_hook,
        strict_map_key=False,
    )


def extract_text(msg):
    # Already normalized (input)
    if isinstance(msg, dict) and "role" in msg and "content" in msg:
        return msg["role"].upper(), msg["content"]

    # Proper LangChain message
    if hasattr(msg, "content"):
        role = msg.__class__.__name__.replace("Message", "").upper()
        return role, msg.content

    # Raw binary fallback
    if isinstance(msg, dict) and "__raw_message__" in msg:
        raw = msg["__raw_message__"]
        text = raw.decode("utf-8", errors="ignore")
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
        return "UNKNOWN", text.strip()

    return None, None


def analyze_checkpoint_db():
    conn = sqlite3.connect(str(CHECKPOINT_DB))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT thread_id, checkpoint, metadata FROM checkpoints ORDER BY rowid"
    )
    rows = cursor.fetchall()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for thread_id, checkpoint, metadata in rows:
            try:
                decoded = decode_checkpoint(checkpoint)

                messages = decoded.get("channel_values", {}).get("messages", [])

                if not messages:
                    continue

                meta = json.loads(metadata.decode("utf-8"))
                header = (
                    f"THREAD {thread_id} | "
                    f"source={meta.get('source')} | "
                    f"step={meta.get('step')}\n"
                )

                f.write(header)

                for msg in messages:
                    role, text = extract_text(msg)
                    if text:
                        f.write(f"{role}: {text}\n")

                f.write("-" * 60 + "\n\n")

            except Exception as e:
                f.write(f"THREAD {thread_id} | ERROR: {e}\n")
                f.write("-" * 60 + "\n\n")

    conn.close()
    print(f"✅ Text written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    analyze_human_logs()
