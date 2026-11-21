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


HUMAN_LOGS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS human_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT,
        timestamp TEXT,
        actor TEXT,
        message TEXT,
        metadata TEXT
    )
"""

# How far each long-term memory pipeline has processed human_logs, by row id.
# Row ids only grow, so "id > last_log_id" is exactly the unprocessed logs of every
# thread, and a pipeline moves its mark only after a batch is stored successfully.
MEMORY_PROGRESS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS memory_progress (
        pipeline TEXT PRIMARY KEY,
        last_log_id INTEGER NOT NULL,
        updated_at TEXT
    )
"""

MEMORY_PIPELINES = ("knowledge_graph", "episodic_rag")


async def log_event(thread_id: str, actor: str, message: str, metadata: dict = None):
    """Saves a human-readable log entry to a separate table."""
    async with aiosqlite.connect(MEMORY_DB) as db:
        await db.execute(HUMAN_LOGS_SCHEMA)
        await db.execute(
            "INSERT INTO human_logs (thread_id, timestamp, actor, message, metadata) VALUES (?, ?, ?, ?, ?)",
            (
                thread_id,
                datetime.now().isoformat(),
                actor,
                message,
                # Real JSON, so the json_extract filters in the memory queries can read it.
                json.dumps(metadata or {}, default=str),
            ),
        )
        await db.commit()


# Standing instructions: how the user wants things done from now on ("always ask before
# sending an email"). Kept as plain rows, not in the knowledge graph (which holds facts
# about people, projects and organizations), and shown to every agent on every message.
INSTRUCTIONS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS user_instructions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        created_at TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
    )
"""


async def add_instruction(text: str) -> int:
    """Store a standing instruction; returns its id."""
    async with aiosqlite.connect(MEMORY_DB) as db:
        await db.execute(INSTRUCTIONS_SCHEMA)
        cursor = await db.execute(
            "INSERT INTO user_instructions (text, created_at) VALUES (?, ?)",
            (text.strip(), datetime.now().isoformat(timespec="seconds")),
        )
        await db.commit()
        return cursor.lastrowid


async def list_instructions() -> list:
    """Active standing instructions, oldest first, as (id, text, created_at)."""
    async with aiosqlite.connect(MEMORY_DB) as db:
        await db.execute(INSTRUCTIONS_SCHEMA)
        async with db.execute(
            "SELECT id, text, created_at FROM user_instructions WHERE active = 1 ORDER BY id"
        ) as cursor:
            return await cursor.fetchall()


async def remove_instruction(instruction_id: int) -> bool:
    """Stop applying a standing instruction (kept in the table, marked inactive)."""
    async with aiosqlite.connect(MEMORY_DB) as db:
        await db.execute(INSTRUCTIONS_SCHEMA)
        cursor = await db.execute(
            "UPDATE user_instructions SET active = 0 WHERE id = ? AND active = 1",
            (int(instruction_id),),
        )
        await db.commit()
        return cursor.rowcount > 0


async def get_memory_progress(db, pipeline: str) -> int:
    """Last human_logs id the pipeline has stored. Creates the record on first use.

    First use on an existing memory.db starts at 0: every log not yet in the progress
    table is processed once. Call init_memory_progress() first to skip logs an older
    version of the memory layer already stored.
    """
    await db.execute(HUMAN_LOGS_SCHEMA)
    await db.execute(MEMORY_PROGRESS_SCHEMA)
    async with db.execute(
        "SELECT last_log_id FROM memory_progress WHERE pipeline = ?", (pipeline,)
    ) as cursor:
        row = await cursor.fetchone()
    if row is not None:
        return row[0]
    await db.execute(
        "INSERT INTO memory_progress (pipeline, last_log_id, updated_at) VALUES (?, 0, ?)",
        (pipeline, datetime.now().isoformat()),
    )
    await db.commit()
    return 0


async def set_memory_progress(db, pipeline: str, last_log_id: int):
    """Record that the pipeline has stored every log up to last_log_id."""
    await db.execute(MEMORY_PROGRESS_SCHEMA)
    await db.execute(
        """
        INSERT INTO memory_progress (pipeline, last_log_id, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(pipeline) DO UPDATE SET
            last_log_id = MAX(last_log_id, excluded.last_log_id),
            updated_at = excluded.updated_at
        """,
        (pipeline, last_log_id, datetime.now().isoformat()),
    )
    await db.commit()


async def init_memory_progress(processed_until: float | None):
    """One-time migration from the old per-thread memory timestamps.

    The old memory layer kept "processed until" as a timestamp in each thread's graph
    state. For a pipeline with no progress record yet, mark every log written up to that
    time as done, so upgrading does not index the same history twice. Does nothing once
    the progress records exist.
    """
    async with aiosqlite.connect(MEMORY_DB) as db:
        await db.execute(HUMAN_LOGS_SCHEMA)
        await db.execute(MEMORY_PROGRESS_SCHEMA)
        last_log_id = 0
        if processed_until:
            async with db.execute(
                "SELECT COALESCE(MAX(id), 0) FROM human_logs WHERE timestamp <= ?",
                (datetime.fromtimestamp(processed_until).isoformat(),),
            ) as cursor:
                last_log_id = (await cursor.fetchone())[0]
        for pipeline in MEMORY_PIPELINES:
            await db.execute(
                "INSERT OR IGNORE INTO memory_progress (pipeline, last_log_id, updated_at) VALUES (?, ?, ?)",
                (pipeline, last_log_id, datetime.now().isoformat()),
            )
        await db.commit()


async def has_pending_memory(logged_before: datetime | None = None) -> bool:
    """True if some pipeline has logs it has not stored yet.

    logged_before: only count logs written before this time (e.g. before today).
    """
    async with aiosqlite.connect(MEMORY_DB) as db:
        progress = [await get_memory_progress(db, p) for p in MEMORY_PIPELINES]
        query = "SELECT 1 FROM human_logs WHERE id > ?"
        params = [min(progress)]
        if logged_before is not None:
            query += " AND timestamp < ?"
            params.append(logged_before.isoformat())
        async with db.execute(query + " LIMIT 1", params) as cursor:
            return await cursor.fetchone() is not None


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

                # Metadata is a JSON string (older rows: a stringified dict)
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
