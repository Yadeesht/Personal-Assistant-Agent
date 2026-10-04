"""
Tests for the long-term memory layer (knowledge graph + episodic RAG).

Run with:
    pytest tests/test_memory_layer.py -v

Memory is saved from human_logs into both pipelines when a chat ends (/new, /thread,
exit), at startup, and on the first turn of a new day. Each pipeline keeps a progress
mark (last human_logs id stored) in the memory_progress table and moves it only after a
batch is stored. These tests run the real flush code against a temporary memory.db;
only the model-backed parts (knowledge-graph extraction, embeddings, the Qdrant index)
are replaced with fakes.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite
import numpy as np
import pytest

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app_tools.tools.rag_tools as rag_tools  # noqa: E402
import core.agent as agent  # noqa: E402
import core.state as state_mod  # noqa: E402
import utils.memory_manager as memory_manager  # noqa: E402
from rag.episodic_rag import EpisodicRAG  # noqa: E402

LONG = "I am planning the quarterly offsite with Raajan from the college club next month. " * 3


class FakeKG:
    """Stands in for rag.knowledge_graph.KnowledgeGraph (no model, no kuzu)."""

    def __init__(self):
        self.extraction_calls = []
        self.entities = []
        self.fail_with = None
        self.invalid_json = False

    def generate_entity_relation(self, text, prompt=None):
        if self.fail_with:
            raise self.fail_with
        self.extraction_calls.append(text)
        if self.invalid_json:
            return None
        return {
            "candidates": {
                "entities": [{"id": "Raajan", "type": "Person", "search_keywords": []}],
                "relationships": [],
            }
        }

    def search_similar_node(self, entities):
        return []

    def validate_entity_relation(self, existing, candidates, prompt=None):
        return {
            "resolution": {
                "entities": [
                    {"action": "CREATE", **e} for e in candidates["candidates"]["entities"]
                ],
                "relationships": [],
            }
        }

    def add_entity(self, **kwargs):
        self.entities.append(kwargs["node_id"])
        self.learned_from = kwargs.get("learned_from")

    def add_relationship(self, **kwargs):
        pass

    def modify_relationship(self, **kwargs):
        pass

    def visualize(self):
        pass


@pytest.fixture
def memory(tmp_path, monkeypatch):
    """Point the memory layer at a temp DB and fake the model-backed parts."""
    db_path = tmp_path / "memory.db"
    monkeypatch.setattr(memory_manager, "MEMORY_DB", db_path)
    monkeypatch.setattr(agent, "MEMORY_DB", db_path)

    kg = FakeKG()
    monkeypatch.setattr(rag_tools, "get_kg_instance", lambda: kg)
    monkeypatch.setattr(rag_tools, "get_profile", lambda: ("", set()))

    indexed = []
    index_ok = {"value": True}

    def fake_index(self, chunks):
        if not index_ok["value"]:
            return False
        indexed.extend(chunks)
        return True

    monkeypatch.setattr(EpisodicRAG, "index_creation", fake_index)
    monkeypatch.setattr(
        EpisodicRAG, "_embedding_chunk", lambda self, chunk: np.zeros(768)
    )

    class Memory:
        pass

    m = Memory()
    m.db_path, m.kg, m.indexed, m.index_ok = db_path, kg, indexed, index_ok
    return m


def log_turn(thread_id, user_text=LONG, reply="Noted, SIR. " + LONG):
    async def _log():
        await memory_manager.log_event(thread_id, "Human_node", user_text, {})
        await memory_manager.log_event(
            thread_id, "supervisor", "route_to_agent(...)", {"type": "tool_call"}
        )
        await memory_manager.log_event(
            thread_id, "supervisor", f"Direct response: {reply}", {"type": "content"}
        )

    asyncio.run(_log())


def progress(db_path):
    async def _read():
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT pipeline, last_log_id FROM memory_progress"
            ) as cursor:
                return dict(await cursor.fetchall())

    return asyncio.run(_read())


def flush(**kwargs):
    return asyncio.run(agent.flush_memory("test", **kwargs))


def test_flush_stores_all_threads_once_and_moves_progress(memory):
    log_turn("1")
    log_turn("2")

    assert flush() == {"knowledge_graph": True, "episodic_rag": True}
    assert progress(memory.db_path) == {"knowledge_graph": 6, "episodic_rag": 6}

    # One extraction batch per thread, tool calls left out.
    assert len(memory.kg.extraction_calls) == 2
    assert all("route_to_agent" not in text for text in memory.kg.extraction_calls)
    # One episode per thread: episodes never span two threads.
    assert len(memory.indexed) == 2

    # Nothing new: no model calls and no re-indexing (the old per-thread timestamps
    # re-indexed other threads' logs as duplicate chunks).
    assert flush() == {"knowledge_graph": True, "episodic_rag": True}
    assert len(memory.kg.extraction_calls) == 2
    assert len(memory.indexed) == 2

    log_turn("3")
    flush()
    assert len(memory.kg.extraction_calls) == 3
    assert len(memory.indexed) == 3
    assert progress(memory.db_path) == {"knowledge_graph": 9, "episodic_rag": 9}


def test_failed_pipeline_keeps_its_logs_for_the_next_run(memory):
    log_turn("1")
    memory.kg.fail_with = ConnectionError("model unreachable")
    memory.index_ok["value"] = False

    assert flush() == {"knowledge_graph": False, "episodic_rag": False}
    assert progress(memory.db_path) == {"knowledge_graph": 0, "episodic_rag": 0}
    assert asyncio.run(memory_manager.has_pending_memory())

    memory.kg.fail_with = None
    memory.index_ok["value"] = True
    assert flush() == {"knowledge_graph": True, "episodic_rag": True}
    assert memory.kg.entities == ["Raajan"]
    assert len(memory.indexed) == 1
    assert not asyncio.run(memory_manager.has_pending_memory())


def test_kg_failure_midway_keeps_the_batches_already_stored(memory):
    log_turn("1")
    log_turn("2")

    real_generate = memory.kg.generate_entity_relation

    def fail_on_second_thread(text, prompt=None):
        if len(memory.kg.extraction_calls) == 1:
            raise ConnectionError("model unreachable")
        return real_generate(text, prompt)

    memory.kg.generate_entity_relation = fail_on_second_thread
    assert flush()["knowledge_graph"] is False
    # Thread 1's batch (logs 1-3) is stored; thread 2 waits for the next run.
    assert progress(memory.db_path)["knowledge_graph"] == 3

    memory.kg.generate_entity_relation = real_generate
    assert flush()["knowledge_graph"] is True
    assert len(memory.kg.extraction_calls) == 2
    assert progress(memory.db_path)["knowledge_graph"] == 6


def test_invalid_json_twice_skips_the_batch_instead_of_blocking(memory):
    log_turn("1")
    memory.kg.invalid_json = True

    assert flush()["knowledge_graph"] is True
    assert len(memory.kg.extraction_calls) == 2  # retried once, then skipped
    assert memory.kg.entities == []
    assert progress(memory.db_path)["knowledge_graph"] == 3


def test_logged_before_leaves_newer_logs_for_later(memory):
    log_turn("1")
    flush(logged_before=datetime.now() - timedelta(days=1))
    assert progress(memory.db_path) == {"knowledge_graph": 0, "episodic_rag": 0}
    assert memory.kg.extraction_calls == []


def test_migration_keeps_logs_the_old_memory_layer_already_stored(memory):
    log_turn("1")
    old_watermark = datetime.now().timestamp() + 1
    log_turn("1")

    async def _set_old_rows_back():
        # The first turn happened before the old watermark, the second after it.
        async with aiosqlite.connect(memory.db_path) as db:
            past = datetime.fromtimestamp(old_watermark - 60).isoformat()
            future = datetime.fromtimestamp(old_watermark + 60).isoformat()
            await db.execute("UPDATE human_logs SET timestamp = ? WHERE id <= 3", (past,))
            await db.execute("UPDATE human_logs SET timestamp = ? WHERE id > 3", (future,))
            await db.commit()

    asyncio.run(_set_old_rows_back())
    asyncio.run(memory_manager.init_memory_progress(old_watermark))
    assert progress(memory.db_path) == {"knowledge_graph": 3, "episodic_rag": 3}

    # Runs once: an existing progress record is never moved back.
    asyncio.run(memory_manager.init_memory_progress(None))
    assert progress(memory.db_path) == {"knowledge_graph": 3, "episodic_rag": 3}

    flush()
    assert len(memory.kg.extraction_calls) == 1  # only the second turn


def test_route_start_saves_earlier_days_once_per_day(memory):
    log_turn("1")

    async def _age_logs():
        async with aiosqlite.connect(memory.db_path) as db:
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            await db.execute("UPDATE human_logs SET timestamp = ?", (yesterday,))
            await db.commit()

    asyncio.run(_age_logs())
    asyncio.run(memory_manager.init_memory_progress(None))

    fresh = {"messages": [], "current_agent": "supervisor"}
    assert asyncio.run(state_mod.route_start(fresh)) == "memory_update_node"

    tried_today = {**fresh, "last_memory_timestamp": datetime.now().timestamp()}
    assert asyncio.run(state_mod.route_start(tried_today)) == "supervisor"

    flush()
    assert asyncio.run(state_mod.route_start(fresh)) == "supervisor"


# ===========================================================================
# What the workers find reaches the knowledge graph, with where it came from.
# ===========================================================================


def test_worker_results_reach_the_knowledge_graph_with_their_source(memory):
    async def _log():
        await memory_manager.log_event("7", "Human_node", "Find Raajan's email address please", {})
        await memory_manager.log_event(
            "7", "communication_agent", "__Tool Action__: Used work_completion ...", {"type": "tool_call"}
        )
        await memory_manager.log_event(
            "7", "communication_agent", "Raajan's email is raajan@club.org", {"type": "handoff_result"}
        )
        await memory_manager.log_event(
            "7", "code_agent", "EXECUTION RESULT: ...", {"type": "code_output"}
        )
        # A kind of log row that does not exist yet must not reach the graph either.
        await memory_manager.log_event(
            "7", "communication_agent", "raw tool payload {...}", {"type": "tool_debug"}
        )
        await memory_manager.log_event(
            "7", "supervisor", "Direct response: Raajan's email is raajan@club.org, SIR.", {"type": "content"}
        )

    asyncio.run(_log())
    flush()

    (text,) = memory.kg.extraction_calls
    assert text.splitlines() == [
        "Yadeesh (user): Find Raajan's email address please",
        "Gmail agent (found in Yadeesh's mailbox): Raajan's email is raajan@club.org",
        "Assistant: Direct response: Raajan's email is raajan@club.org, SIR.",
    ]
    assert memory.kg.learned_from == f"conversation (thread 7) on {datetime.now():%Y-%m-%d}"


def test_worker_node_logs_its_work_completion_result(memory, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    monkeypatch.setattr(rag_tools, "recall_memory", lambda query: "")

    class FakeLLM:
        async def ainvoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "work_completion",
                        "args": {"message": "Raajan's email is raajan@club.org"},
                        "id": "call_1",
                    }
                ],
            )

    node = agent.agent_node_factory(FakeLLM(), "prompt {current_time}", "communication_agent")
    state = {
        "messages": [HumanMessage(content="Find Raajan's email address please", id="m1")],
        "communication_messages": [
            HumanMessage(content="[Handoff from supervisor] ...", name="supervisor")
        ],
    }
    asyncio.run(node(state, {"configurable": {"thread_id": "7"}}))

    async def _rows():
        async with aiosqlite.connect(memory.db_path) as db:
            async with db.execute(
                "SELECT thread_id, actor, message, json_extract(metadata, '$.type') FROM human_logs"
            ) as cursor:
                return await cursor.fetchall()

    rows = asyncio.run(_rows())
    assert ("7", "communication_agent", "Raajan's email is raajan@club.org", "handoff_result") in rows


# ===========================================================================
# Saving runs in the background; exit waits for it.
# ===========================================================================


def test_background_saver_queues_one_more_run_while_busy(monkeypatch):
    import threading
    import time

    calls, release = [], threading.Event()

    async def slow_flush(reason="", logged_before=None):
        calls.append(reason)
        if reason == "first":
            release.wait(5)
        return {"knowledge_graph": True, "episodic_rag": True}

    monkeypatch.setattr(agent, "flush_memory", slow_flush)
    saver = agent.BackgroundMemorySaver()

    saver.request("first")
    time.sleep(0.1)
    assert saver.running
    saver.request("second")
    saver.request("third")  # replaces the queued run; one extra run covers everything

    release.set()
    assert saver.wait(5)
    assert calls == ["first", "third"]
    assert saver.last_result == {"knowledge_graph": True, "episodic_rag": True}
    assert not saver.running


def test_background_saver_reports_a_failed_save(monkeypatch):
    async def broken_flush(reason="", logged_before=None):
        raise RuntimeError("disk full")

    monkeypatch.setattr(agent, "flush_memory", broken_flush)
    saver = agent.BackgroundMemorySaver()
    saver.request("exit")
    assert saver.wait(5)
    assert saver.last_result == {"knowledge_graph": False, "episodic_rag": False}


def test_background_save_really_stores_memory(memory):
    log_turn("1")
    saver = agent.BackgroundMemorySaver()
    saver.request("/new")
    assert saver.wait(10)
    assert saver.last_result == {"knowledge_graph": True, "episodic_rag": True}
    assert progress(memory.db_path) == {"knowledge_graph": 3, "episodic_rag": 3}
    assert memory.kg.entities == ["Raajan"]


# ===========================================================================
# Recall: related memory is looked up once per turn and shown to the agents.
# ===========================================================================


def test_recall_is_looked_up_once_per_turn_and_shown_to_the_supervisor(memory, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    lookups = []

    def fake_recall(query, exclude_entities=()):
        lookups.append(query)
        return "Known facts:\n- Raajan (Person): email raajan@club.org"

    monkeypatch.setattr(rag_tools, "recall_memory", fake_recall)

    seen = {}

    class FakeLLM:
        async def ainvoke(self, messages):
            seen["messages"] = messages
            return AIMessage(content="Raajan's email is raajan@club.org, SIR.")

    node = agent.supervisor_node_factory(FakeLLM(), "You are JARVIS. {current_time}")
    user = HumanMessage(content="What is Raajan's email address?", id="m1")
    state = {"messages": [user], "supervisor_messages": [user]}

    update = asyncio.run(node(state, {"configurable": {"thread_id": "7"}}))
    assert lookups == ["What is Raajan's email address?"]
    assert update["recalled_for"] == "m1"
    assert "raajan@club.org" in update["recalled_memory"]
    system_texts = [m.content for m in seen["messages"] if m.type == "system"]
    assert system_texts[0].startswith("You are JARVIS.")
    assert system_texts[1].startswith("Recalled memory") and "raajan@club.org" in system_texts[1]

    # Later steps of the same turn reuse the lookup.
    asyncio.run(node({**state, **update}, {"configurable": {"thread_id": "7"}}))
    assert len(lookups) == 1


def test_recall_memory_finds_named_and_related_facts_only_from_the_graph(monkeypatch):
    import pandas

    class KG:
        def count_entities(self):
            return 3

        def entities_named_in(self, text):
            return ["Raajan"] if "raajan" in text.lower() else []

        def entity_facts(self, ids):
            return [
                {"id": "Raajan", "type": "Person", "description": "Club collaborator; email raajan@club.org",
                 "updated_at": "2026-10-04T10:00:00"}
            ] if "Raajan" in ids else []

        def find_similar_with_expansion(self, query):
            return pandas.DataFrame(
                [
                    {"id": "Offsite 2026", "type": "Event", "description": "Quarterly offsite in Goa",
                     "updated_at": "", "hops": 0, "base_score": 0.72, "Relations": "DIRECT_HIT"},
                    {"id": "Weak Match", "type": "Concept", "description": "barely related",
                     "updated_at": "", "hops": 0, "base_score": 0.4, "Relations": "DIRECT_HIT"},
                    {"id": "Yadeesh", "type": "Person", "description": "the user",
                     "updated_at": "", "hops": 0, "base_score": 0.9, "Relations": "DIRECT_HIT"},
                ]
            )

        def relationships_of(self, ids, limit):
            return [{"source": "Raajan", "relation_type": "ORGANIZES", "target": "Offsite 2026"}]

    def no_episodic():
        raise AssertionError("past conversations must not be looked up automatically")

    monkeypatch.setattr(rag_tools, "get_kg_instance", lambda: KG())
    monkeypatch.setattr(rag_tools, "get_rag_instance", no_episodic)
    monkeypatch.setattr(rag_tools, "MEMORY_RECALL", True)

    text = rag_tools.recall_memory("email Raajan about the offsite", exclude_entities={"Yadeesh"})
    # Named in the message: always shown. Found by meaning: only above the threshold.
    assert "- Raajan (Person): Club collaborator; email raajan@club.org (stored 2026-10-04)" in text
    assert "- Offsite 2026 (Event): Quarterly offsite in Goa" in text
    assert "Weak Match" not in text
    assert "- Yadeesh" not in text  # already in the profile
    assert "Connections: Raajan ORGANIZES Offsite 2026" in text
    assert "Earlier conversations" not in text

    # A short reply still finds an entity it names, but runs no similarity search.
    short = rag_tools.recall_memory("ask Raajan")
    assert "- Raajan (Person)" in short and "- Offsite 2026" not in short
    assert rag_tools.recall_memory("yes ok") == ""

    monkeypatch.setattr(rag_tools, "MEMORY_RECALL", False)
    assert rag_tools.recall_memory("email Raajan about the offsite") == ""


def test_past_conversations_are_looked_up_on_request_with_their_date(monkeypatch):
    class RAG:
        def retrieve_chunks(self, query, conditions, top_k):
            return [{"context": "User: plan the offsite with Raajan", "score": 0.82,
                     "type": "raw_chunk", "timestamp": "2026-09-12T18:30:00", "thread_id": "4"}]

    monkeypatch.setattr(rag_tools, "get_rag_instance", lambda: RAG())
    hits = asyncio.run(rag_tools.retrieve_relevant_chunks("when did we plan the offsite"))
    assert hits[0]["timestamp"] == "2026-09-12T18:30:00" and hits[0]["thread_id"] == "4"


# ===========================================================================
# Keeping facts current, against a real (temporary) kuzu graph. Only the embedding
# model is replaced, by a fixed vector per text.
# ===========================================================================


def _fake_embedding(text, is_query=False):
    rng = np.random.default_rng(sum(text.encode()) * 7919 % (2**32))
    vector = rng.normal(size=384)
    return (vector / np.linalg.norm(vector)).tolist()


def _relations(kg):
    rows = kg.conn.execute(
        "MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) RETURN a.id, r.relation_type, b.id"
    ).get_as_df()
    return sorted(tuple(row) for row in rows.itertuples(index=False))


def test_updating_an_entity_keeps_its_relationships_and_records_freshness(tmp_path):
    from rag.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg" / "graph.db"))
    kg._compute_embedding = _fake_embedding

    kg.add_entity("Yadeesh", "Person", "yadeesh", "The user", learned_from="told directly")
    kg.add_entity("Raajan", "Person", "raajan", "Club collaborator; email old@club.org")
    kg.add_entity("College Club", "Organization", "club", "Student club")
    kg.add_relationship("Yadeesh", "Raajan", "WORKS_WITH")
    kg.add_relationship("Raajan", "College Club", "MEMBER_OF")
    kg.add_relationship("Raajan", "College Club", "MEMBER_OF")  # same fact again: no duplicate

    before = _relations(kg)
    assert before == [
        ("Raajan", "MEMBER_OF", "College Club"),
        ("Yadeesh", "WORKS_WITH", "Raajan"),
    ]

    # A newer fact about Raajan replaces the old description; both of Raajan's
    # relationships (one incoming, one outgoing) survive the update.
    kg.add_entity(
        "Raajan",
        "Person",
        "raajan",
        "Club collaborator; email new@club.org",
        learned_from="conversation (thread 7) on 2026-10-04",
    )
    assert _relations(kg) == before

    row = kg.conn.execute(
        "MATCH (n:Entity {id: 'Raajan'}) RETURN n.description, n.updated_at, n.learned_from"
    ).get_next()
    assert row[0] == "Club collaborator; email new@club.org"
    assert row[1].startswith(f"{datetime.now():%Y-%m-%d}")
    assert row[2] == "conversation (thread 7) on 2026-10-04"

    # Lookups report how fresh each fact is.
    found = kg.find_similar_with_expansion("Raajan")
    assert "updated_at" in found.columns
    assert kg.count_entities() == 3
    kg.close()


def test_existing_graph_gets_the_freshness_columns(tmp_path):
    import kuzu

    from rag.knowledge_graph import KnowledgeGraph

    path = tmp_path / "kg" / "graph.db"
    path.parent.mkdir(parents=True)
    old_db = kuzu.Database(str(path))
    old = kuzu.Connection(old_db)
    old.execute(
        "CREATE NODE TABLE Entity(id STRING, type STRING, search_keywords STRING, "
        "description STRING, embedding FLOAT[384], PRIMARY KEY(id))"
    )
    old.execute("CREATE REL TABLE RELATED_TO(FROM Entity TO Entity, relation_type STRING)")
    old.execute(
        "CREATE (n:Entity {id: 'Raajan', type: 'Person', search_keywords: '', description: 'old fact'})"
    )
    old.close()
    old_db.close()

    kg = KnowledgeGraph(db_path=str(path))
    row = kg.conn.execute(
        "MATCH (n:Entity {id: 'Raajan'}) RETURN n.description, n.updated_at, n.learned_from"
    ).get_next()
    assert row == ["old fact", "", ""]
    kg.close()


# ===========================================================================
# Short-term memory: the summarizer cuts every message list at turn boundaries,
# so no tool result is left without its tool call.
# ===========================================================================


@pytest.fixture
def small_summary_limits(monkeypatch):
    """The summarizer tests build threads of ~20k tokens; pin the limits they assume."""
    monkeypatch.setattr(state_mod, "SUMMARY_TRIGGER_TOKENS", 8000)
    monkeypatch.setattr(state_mod, "SUMMARY_KEEP_TOKENS", 4000)


def _run_turns(turns, body_words=400):
    """Several supervisor -> Gmail agent -> supervisor turns, built with the real
    route_to_agent / work_completion tools and the add_messages reducer."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.graph.message import add_messages
    from langgraph.types import Command

    channels = ["messages", "supervisor_messages", "communication_messages"]
    state = {c: [] for c in channels}
    state["current_agent"] = "supervisor"

    def apply(update):
        if isinstance(update, Command):
            update = update.update
        for key, value in update.items():
            state[key] = add_messages(state[key], value) if key in channels else value

    body = "Email body text. " * body_words
    for turn in range(turns):
        user = HumanMessage(content=f"Turn {turn}: summarise my latest emails from Ravi")
        apply({"messages": [user], "supervisor_messages": [user]})
        sup = AIMessage(content="", name="supervisor", tool_calls=[
            {"name": "route_to_agent", "args": {"agent": "communication_agent"}, "id": f"r{turn}"}])
        apply({"messages": [sup], "supervisor_messages": [sup]})
        # The tools get only the state they read (the latest user message, the active agent).
        apply(agent.route_to_agent.invoke({"type": "tool_call", "id": f"r{turn}", "name": "route_to_agent",
                                           "args": {"agent": "communication_agent",
                                                    "state": {"messages": [user]}}}))
        call = AIMessage(content="", name="communication_agent", tool_calls=[
            {"name": "search_emails", "args": {}, "id": f"s{turn}"}])
        apply({"messages": [call], "communication_messages": [call]})
        result = ToolMessage(content=body, tool_call_id=f"s{turn}")
        apply({"messages": [result], "communication_messages": [result]})
        done = AIMessage(content="", name="communication_agent", tool_calls=[
            {"name": "work_completion", "args": {"message": f"Summary {turn}: " + body}, "id": f"w{turn}"}])
        apply({"messages": [done], "communication_messages": [done]})
        state["current_agent"] = "communication_agent"
        apply(agent.work_completion.invoke({"type": "tool_call", "id": f"w{turn}", "name": "work_completion",
                                            "args": {"message": f"Summary {turn}: " + body,
                                                     "state": {"current_agent": "communication_agent"}}}))
        final = AIMessage(content=f"Here is the summary for turn {turn}, SIR.", name="supervisor")
        apply({"messages": [final], "supervisor_messages": [final]})
    return state, apply, channels


def _assert_tool_pairs_intact(messages):
    from langchain_core.messages import AIMessage, ToolMessage

    called, answered = set(), set()
    for m in messages:
        if isinstance(m, AIMessage):
            called.update(tc["id"] for tc in m.tool_calls or [])
        if isinstance(m, ToolMessage):
            assert m.tool_call_id in called, f"tool result {m.tool_call_id} without its call"
            answered.add(m.tool_call_id)
    assert called == answered, f"tool calls without results: {called - answered}"


def test_summarizer_keeps_tool_calls_with_their_results_in_every_list(memory, monkeypatch, small_summary_limits):
    from langchain_core.messages import AIMessage, HumanMessage

    state, apply, channels = _run_turns(8)
    assert set(state_mod.plan_summary(state)) == set(channels)  # all three are too long

    prompts = []

    class FakeLLM:
        async def ainvoke(self, messages):
            prompts.append(messages[-1].content)
            return AIMessage(content="### Active Goals\nSummaries of Ravi's emails")

    monkeypatch.setattr(agent, "build_llm", lambda *a, **k: FakeLLM())
    apply(asyncio.run(agent.summerizer_node(state, {"configurable": {"thread_id": "7"}})))

    assert state["summary"].startswith("### Active Goals")
    for channel in channels:
        kept = state[channel]
        assert isinstance(kept[0], HumanMessage), channel  # cut at a turn boundary
        _assert_tool_pairs_intact(kept)
        assert agent.count_tokens(kept) <= state_mod.SUMMARY_TRIGGER_TOKENS
    # The latest turn is always kept.
    assert any("Turn 7" in str(m.content) for m in state["messages"])

    # The summarizer read a transcript, not raw message objects, including the handoffs
    # that live only in the agents' own lists.
    (prompt,) = prompts
    assert "User: Turn 0: summarise my latest emails from Ravi" in prompt
    assert "supervisor called route_to_agent" in prompt
    assert "Handoffs between agents in these turns:" in prompt
    assert "additional_kwargs" not in prompt

    # Nothing is left to condense, so the next turn does not summarize again.
    assert state_mod.plan_summary(state) == {}


def test_summarizer_failure_removes_nothing(memory, monkeypatch, small_summary_limits):
    state, apply, channels = _run_turns(8)
    before = {c: len(state[c]) for c in channels}

    class BrokenLLM:
        async def ainvoke(self, messages):
            raise ConnectionError("model unreachable")

    monkeypatch.setattr(agent, "build_llm", lambda *a, **k: BrokenLLM())
    assert asyncio.run(agent.summerizer_node(state, {"configurable": {"thread_id": "7"}})) == {}
    assert {c: len(state[c]) for c in channels} == before


def test_tool_results_without_their_call_are_dropped_before_the_model_call():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    messages = [
        ToolMessage(content="routed", tool_call_id="gone"),  # its call was summarized away
        HumanMessage(content="next request"),
        AIMessage(content="", tool_calls=[{"name": "search_emails", "args": {}, "id": "ok"}]),
        ToolMessage(content="3 emails", tool_call_id="ok"),
    ]
    cleaned = agent.clean_unmatched_tool_calls(messages)
    assert [type(m).__name__ for m in cleaned] == ["HumanMessage", "AIMessage", "ToolMessage"]


def test_a_long_thread_is_summarized_then_resumes_with_the_active_worker(memory, small_summary_limits):
    state, _apply, _channels = _run_turns(8)
    state["current_agent"] = "communication_agent"  # the worker asked the user a question
    assert asyncio.run(state_mod.route_start(state)) == "summerizer_node"
    assert state_mod.route_to_active_agent(state) == "communication_agent"

    state["current_agent"] = "supervisor"
    assert state_mod.route_to_active_agent(state) == "supervisor"


def test_workers_see_the_profile_recall_and_thread_summary(memory, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    monkeypatch.setattr(rag_tools, "get_profile", lambda: ("- Yadeesh (Person): the user", {"Yadeesh"}))
    monkeypatch.setattr(rag_tools, "recall_memory", lambda *a, **k: "Known facts:\n- Ravi (Person): manager")
    seen = {}

    class FakeLLM:
        async def ainvoke(self, messages):
            seen["messages"] = messages
            return AIMessage(content="Which Ravi do you mean, SIR?")

    node = agent.agent_node_factory(FakeLLM(), "Gmail agent {current_time}", "communication_agent")
    state = {
        "messages": [HumanMessage(content="Email Ravi the report", id="m1")],
        "communication_messages": [HumanMessage(content="[Handoff from supervisor] ...", name="supervisor")],
        "summary": "### Active Goals\nSend the Q3 report",
    }
    asyncio.run(node(state, {"configurable": {"thread_id": "7"}}))

    system = [m.content for m in seen["messages"] if m.type == "system"]
    assert system[0].startswith("Gmail agent")
    assert "About the user:\n- Yadeesh (Person): the user" in system[1]
    assert "Known facts:\n- Ravi (Person): manager" in system[1]
    assert system[2] == "Conversation Summary of previous messages:\n### Active Goals\nSend the Q3 report"


def test_supervisor_replies_are_logged_in_full(memory, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    monkeypatch.setattr(rag_tools, "recall_memory", lambda *a, **k: "")
    reply = "Here is everything, SIR. " + "detail " * 300

    class FakeLLM:
        async def ainvoke(self, messages):
            return AIMessage(content=reply)

    node = agent.supervisor_node_factory(FakeLLM(), "You are JARVIS. {current_time}")
    user = HumanMessage(content="What happened this week?", id="m1")
    asyncio.run(node({"messages": [user], "supervisor_messages": [user]}, {"configurable": {"thread_id": "7"}}))

    async def _logged():
        async with aiosqlite.connect(memory.db_path) as db:
            async with db.execute("SELECT message FROM human_logs WHERE actor = 'supervisor'") as cursor:
                return [row[0] for row in await cursor.fetchall()]

    assert asyncio.run(_logged()) == [f"Direct response: {reply}"]


# ===========================================================================
# Profile and recall lookups against a real (temporary) kuzu graph.
# ===========================================================================


def _small_graph(tmp_path):
    from rag.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg" / "graph.db"))
    kg._compute_embedding = _fake_embedding
    kg.add_entity("Yadeesh", "Person", "yadeesh", "The user; studies at VIT Chennai")
    kg.add_entity("Raajan", "Person", "raajan", "Club collaborator; email raajan@club.org")
    kg.add_entity("College Club", "Organization", "club", "Student club")
    kg.add_entity("Ann", "Person", "ann", "A namesake that must not match inside other words")
    kg.add_relationship("Yadeesh", "Raajan", "WORKS_WITH")
    kg.add_relationship("Raajan", "College Club", "MEMBER_OF")
    kg.add_relationship("College Club", "Yadeesh", "HAS_MEMBER")
    return kg


def test_graph_lookups_for_recall_and_profile(tmp_path):
    kg = _small_graph(tmp_path)

    # Whole-word, any-case name matching ("Ann" is not found inside "planning").
    assert sorted(kg.entities_named_in("Please email raajan about the college club planning")) == [
        "College Club",
        "Raajan",
    ]
    assert [f["id"] for f in kg.entity_facts(["Raajan"])] == ["Raajan"]
    assert {(r["source"], r["relation_type"], r["target"]) for r in kg.relationships_of(["Raajan"])} == {
        ("Yadeesh", "WORKS_WITH", "Raajan"),
        ("Raajan", "MEMBER_OF", "College Club"),
    }

    profile = kg.profile("yadeesh")
    assert profile["entity"]["id"] == "Yadeesh"
    assert {(c["relation_type"], c["id"], c["outgoing"]) for c in profile["connections"]} == {
        ("WORKS_WITH", "Raajan", True),
        ("MEMBER_OF", "College Club", True),  # written as "College Club HAS_MEMBER Yadeesh"
    }
    assert kg.profile("Nobody") == {"entity": None, "connections": []}
    kg.close()


def test_profile_text_is_cached_until_the_graph_changes(tmp_path, monkeypatch):
    kg = _small_graph(tmp_path)
    monkeypatch.setattr(rag_tools, "get_kg_instance", lambda: rag_tools._ThreadSafeKG(kg))
    monkeypatch.setattr(rag_tools, "MEMORY_RECALL", True)
    rag_tools.invalidate_profile()
    try:
        text, ids = rag_tools.get_profile()
        assert text.splitlines()[0].startswith("- Yadeesh (Person): The user; studies at VIT Chennai (stored ")
        assert "- Yadeesh WORKS_WITH Raajan: Raajan (Person) Club collaborator; email raajan@club.org" in text
        assert "- Yadeesh MEMBER_OF College Club: College Club (Organization) Student club" in text
        assert ids == {"Yadeesh", "Raajan", "College Club"}

        kg.add_entity("Priya", "Person", "priya", "Project mentor")
        kg.add_relationship("Priya", "Yadeesh", "works with")
        assert rag_tools.get_profile()[0] == text  # cached
        rag_tools.invalidate_profile()  # what a memory save does after changing the graph
        assert "- Priya WORKS_WITH Yadeesh: Priya (Person) Project mentor" in rag_tools.get_profile()[0]
    finally:
        rag_tools.invalidate_profile()
        kg.close()


def test_look_alikes_include_an_entity_with_the_same_name(tmp_path):
    kg = _small_graph(tmp_path)
    candidates = [{"id": "raajan", "type": "Person", "description": "worded completely differently",
                   "search_keywords": ["x"]}]
    found = kg.search_similar_node(candidates)
    row = found[found["id"] == "Raajan"].iloc[0]
    assert row["similarity_score"] == 1.0
    assert "WORKS_WITH with Yadeesh" in row["relations"] or "MEMBER_OF with College Club" in row["relations"]
    kg.close()


def test_episodic_recall_can_leave_out_the_current_thread(tmp_path, monkeypatch):
    import uuid

    import rag.episodic_rag as episodic

    monkeypatch.setattr(episodic, "EPISODIC_RAG_DB", tmp_path / "episodic")
    (tmp_path / "episodic").mkdir()
    vector = np.ones(768) / np.sqrt(768)
    rag = EpisodicRAG(db_path=str(tmp_path / "memory.db"))
    rag._embedding_chunk = lambda chunk: vector
    assert not rag.has_chunks()

    def chunk(thread, text):
        return {
            "id": str(uuid.uuid4()),
            "content": text,
            "embedding": vector,
            "metadata": {"timestamp": "2026-10-04T10:00:00", "task_id": str(uuid.uuid4()), "part": 1,
                         "total_parts": 1, "actors": ["supervisor"], "prev_id": None, "next_id": None,
                         "thread_id": thread},
        }

    assert rag.index_creation([chunk("1", "from the current thread"), chunk("2", "from an older thread")])
    assert rag.has_chunks()
    hits = rag.retrieve_chunks("offsite", None, 5)
    assert sorted(h["context"] for h in hits) == ["from an older thread", "from the current thread"]
    assert {(h["thread_id"], h["timestamp"]) for h in hits} == {
        ("1", "2026-10-04T10:00:00"),
        ("2", "2026-10-04T10:00:00"),
    }
    others = [h["context"] for h in rag.retrieve_chunks("offsite", {"exclude_thread_id": "1"}, 5)]
    assert others == ["from an older thread"]


# ===========================================================================
# Embedding models are saved crash-safely, and background logs stay off the prompt.
# ===========================================================================


class _FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer (no download)."""

    loads = []
    fail_save = False

    def __init__(self, name):
        _FakeSentenceTransformer.loads.append(str(name))

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config_sentence_transformers.json").write_text("{}")
        if _FakeSentenceTransformer.fail_save:
            raise KeyboardInterrupt("stopped mid-save")
        (path / "model.safetensors").write_text("weights")
        (path / "modules.json").write_text("[]")


def test_a_half_written_model_folder_is_rebuilt(tmp_path, monkeypatch):
    import sentence_transformers

    from utils.helper import load_sentence_model

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    _FakeSentenceTransformer.loads = []
    _FakeSentenceTransformer.fail_save = False

    # What an interrupted save used to leave: the folder exists, the model does not.
    target = tmp_path / "gte-base"
    target.mkdir()
    (target / "config_sentence_transformers.json").write_text("{}")

    load_sentence_model(target, "unsloth/gte-modernbert-base")
    assert _FakeSentenceTransformer.loads == ["unsloth/gte-modernbert-base"]  # fetched again
    assert (target / "modules.json").exists() and (target / "model.safetensors").exists()
    assert not (tmp_path / "gte-base.partial").exists()

    load_sentence_model(target, "unsloth/gte-modernbert-base")
    assert _FakeSentenceTransformer.loads[-1] == str(target)  # complete: loaded locally


def test_a_save_stopped_midway_leaves_no_broken_model(tmp_path, monkeypatch):
    import sentence_transformers

    from utils.helper import load_sentence_model

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    _FakeSentenceTransformer.fail_save = True
    target = tmp_path / "gte-base"
    with pytest.raises(KeyboardInterrupt):
        load_sentence_model(target, "unsloth/gte-modernbert-base")
    assert not target.exists()  # only the .partial folder was written

    _FakeSentenceTransformer.fail_save = False
    load_sentence_model(target, "unsloth/gte-modernbert-base")  # next run recovers
    assert (target / "modules.json").exists()
    assert not (tmp_path / "gte-base.partial").exists()


def test_background_thread_logs_stay_off_the_console(tmp_path):
    import logging

    from rich.logging import RichHandler

    import utils.helper as helper

    root = logging.getLogger()
    console_handler = next(h for h in root.handlers if isinstance(h, RichHandler))
    file_handlers = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]

    def record(level, thread):
        r = logging.LogRecord("rag.episodic_rag", level, __file__, 1, "msg", None, None)
        r.threadName = thread
        return r

    # Routine background messages: file only. Warnings: also on the console.
    assert not console_handler.filter(record(logging.INFO, "memory-saver"))
    assert console_handler.filter(record(logging.WARNING, "memory-saver"))
    assert console_handler.filter(record(logging.INFO, "MainThread"))
    assert file_handlers and all(h.filter(record(logging.INFO, "memory-preload")) for h in file_handlers)
    assert not any(h.filter(record(logging.INFO, "MainThread")) for h in file_handlers)
    # The tests write background logs to a temp file (tests/conftest.py), not data/logs.
    assert helper.BACKGROUND_LOG_FILE == Path(os.environ["MEMORY_LOG_FILE"])
    assert all(Path(h.baseFilename) == helper.BACKGROUND_LOG_FILE for h in file_handlers)


# ===========================================================================
# Standing instructions: saved by the supervisor, shown to every agent each message.
# ===========================================================================


def test_standing_instructions_are_saved_listed_and_dropped(memory):
    saved = asyncio.run(rag_tools.remember_instruction(
        "Before sending any email, show SIR the recipient and full content and wait for approval."
    ))
    assert saved.startswith("Saved standing instruction [1]")
    asyncio.run(rag_tools.remember_instruction("Keep email summaries under 5 bullet points."))
    assert asyncio.run(rag_tools.remember_instruction("   ")) == "Nothing saved: the instruction was empty."

    text = asyncio.run(rag_tools.get_instructions_text())
    assert text == (
        "- [1] Before sending any email, show SIR the recipient and full content and wait for approval.\n"
        "- [2] Keep email summaries under 5 bullet points."
    )

    assert asyncio.run(rag_tools.forget_instruction(2)) == "Standing instruction [2] dropped."
    assert asyncio.run(rag_tools.forget_instruction(2)) == "No active standing instruction [2]."
    assert "[2]" not in asyncio.run(rag_tools.get_instructions_text())


def test_every_agent_sees_the_standing_instructions_first(memory, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    monkeypatch.setattr(rag_tools, "get_profile", lambda: ("- Yadeesh (Person): the user", {"Yadeesh"}))
    monkeypatch.setattr(rag_tools, "recall_memory", lambda *a, **k: "")
    asyncio.run(rag_tools.remember_instruction("Always ask before sending an email."))
    seen = {}

    class FakeLLM:
        async def ainvoke(self, messages):
            seen["messages"] = messages
            return AIMessage(content="ok")

    for factory, args in [
        (agent.supervisor_node_factory, (FakeLLM(), "Supervisor {current_time}")),
        (agent.agent_node_factory, (FakeLLM(), "Gmail agent {current_time}", "communication_agent")),
    ]:
        node = factory(*args)
        user = HumanMessage(content="Email Ravi the report", id="m1")
        asyncio.run(node(
            {"messages": [user], "supervisor_messages": [user], "communication_messages": [user]},
            {"configurable": {"thread_id": "7"}},
        ))
        system = [m.content for m in seen["messages"] if m.type == "system"]
        assert system[1] == (
            "SIR's standing instructions (always follow these unless SIR says otherwise in this conversation):\n"
            "- [1] Always ask before sending an email."
        )
        assert system[2].startswith("Recalled memory") and "About the user:" in system[2]

    monkeypatch.setattr(rag_tools, "MEMORY_RECALL", False)  # e.g. evals
    assert asyncio.run(rag_tools.get_instructions_text()) == ""


# ===========================================================================
# Knowledge-graph quality: allowed types, one entity per name, clean relationships.
# ===========================================================================


def test_invented_types_are_mapped_onto_the_seven_allowed_ones():
    from rag.knowledge_graph import normalize_entity_type

    # Types the model invented in the real graph, and where they belong.
    expected = {
        "Educational Institution": "Organization", "Organization": "Organization",
        "Field of Study": "Concept", "Academic Cohort": "Concept", "Career Role": "Concept",
        "Technology concept": "Concept", "Preparation approach": "Concept",
        "LearningPreference": "Concept", "Information request": "Concept",
        "Platform": "Tool", "Technology": "Tool",
        "Research": "Project", "Project": "Project",
        "Data": "Resource", "Person": "Person", "person": "Person",
        "": "Concept", None: "Concept", "Something odd": "Concept",
    }
    assert {t: normalize_entity_type(t) for t in expected} == expected


def test_relationship_names_are_upper_snake_case():
    from rag.knowledge_graph import normalize_relation_type

    assert normalize_relation_type("worksWith") == "WORKS_WITH"
    assert normalize_relation_type("works with") == "WORKS_WITH"
    assert normalize_relation_type("MEMBER_OF") == "MEMBER_OF"
    assert normalize_relation_type("") == "RELATED_TO"


def test_a_name_in_another_capitalization_updates_the_existing_entity(tmp_path):
    kg = _small_graph(tmp_path)

    kg.add_entity("raajan", "Friend", "raajan, club", "Plays chess on weekends")
    assert kg.count_entities() == 4  # no second Raajan
    row = kg._stored_entity("Raajan")
    assert row["type"] == "Person"  # "Friend" -> Person
    assert row["description"] == "Club collaborator; email raajan@club.org. Plays chess on weekends"
    # Relationships named in any capitalization find the same entity.
    kg.add_relationship("RAAJAN", "ann", "manages")  # stored as "Ann REPORTS_TO Raajan"
    assert ("Ann", "REPORTS_TO", "Raajan") in _relations(kg)
    # And Raajan's earlier relationships survived the update.
    assert ("Yadeesh", "WORKS_WITH", "Raajan") in _relations(kg)
    kg.close()


def test_merging_moves_relationships_without_duplicates(tmp_path):
    kg = _small_graph(tmp_path)
    kg.add_entity("Raajan K", "Person", "raajan k", "Same person, surname initial")
    kg.add_relationship("Raajan K", "College Club", "MEMBER_OF")  # Raajan already has this
    kg.add_relationship("Ann", "Raajan K", "KNOWS")

    assert kg.merge_entities("Raajan", "Raajan K")
    assert kg._stored_entity("Raajan K") is None
    relations = _relations(kg)
    assert relations.count(("Raajan", "MEMBER_OF", "College Club")) == 1
    assert ("Ann", "KNOWS", "Raajan") in relations
    assert "Same person, surname initial" in kg._stored_entity("Raajan")["description"]
    kg.close()


def test_tidy_cleans_a_graph_built_before_the_guards(tmp_path):
    import kuzu

    from rag.knowledge_graph import KnowledgeGraph

    path = tmp_path / "kg" / "graph.db"
    path.parent.mkdir(parents=True)
    kg = KnowledgeGraph(db_path=str(path))
    kg._compute_embedding = _fake_embedding
    # Write the mess directly, the way the old code could store it.
    for node_id, node_type in [("Yadeesh", "Person"), ("AI systems", "Technical Focus"),
                               ("AI Systems", "Field"), ("Internships", "Career Goal"),
                               ("Placement preparation", "Activity"), ("Campus placements", "Career Preparation")]:
        kg.conn.execute(
            "CREATE (n:Entity {id: $id, type: $type, search_keywords: '', description: $d, "
            "embedding: $e, updated_at: '', learned_from: ''})",
            parameters={"id": node_id, "type": node_type, "d": f"about {node_id}", "e": _fake_embedding(node_id)},
        )
    for source, rel, target in [("Yadeesh", "FOCUSES_ON_BUILDING", "AI systems"),
                                ("Yadeesh", "FOCUSES_ON_BUILDING", "AI Systems"),
                                ("Yadeesh", "PURSUING", "Internships"),
                                ("Yadeesh", "PURSUUES", "Internships"),
                                ("Yadeesh", "PREPARES_FOR", "Placement preparation"),
                                ("Yadeesh", "PREPARING_FOR", "Campus placements")]:
        kg._create_relationship(source, target, rel, "")

    report = kg.tidy(
        merges=[("Campus placements", "Placement preparation")],
        relation_renames={"PURSUUES": "PURSUES", "PURSUING": "PURSUES", "PREPARING_FOR": "PREPARES_FOR"},
    )

    ids = sorted(r[0] for r in kg.conn.execute("MATCH (n:Entity) RETURN n.id").get_as_df().values)
    # Equal relationship counts: the name that sorts first is kept ("AI Systems").
    assert ids == ["AI Systems", "Campus placements", "Internships", "Yadeesh"]
    types = {r[0] for r in kg.conn.execute("MATCH (n:Entity) RETURN n.type").get_as_df().values}
    assert types <= {"Person", "Concept"}
    assert sorted(_relations(kg)) == [
        ("Yadeesh", "PREPARES_FOR", "Campus placements"),  # PREPARES_FOR + PREPARING_FOR
        ("Yadeesh", "PREPARES_FOR", "Internships"),  # PURSUING and the typo PURSUUES
        ("Yadeesh", "WORKS_ON", "AI Systems"),  # FOCUSES_ON_BUILDING, both capitalizations
    ]
    # The two kept entities got their types fixed while merging; Internships was retyped.
    assert len(report["merged"]) == 2 and report["retyped"] == 1
    kg.close()


def test_explicitly_stored_facts_use_the_standard_extraction_rules(memory, monkeypatch):
    import config.prompts as prompt_texts

    prompts_seen = []
    real_generate = memory.kg.generate_entity_relation

    def capture(text, prompt=None):
        prompts_seen.append((text, prompt))
        return real_generate(text, prompt)

    memory.kg.generate_entity_relation = capture
    result = asyncio.run(rag_tools.add_information_to_knowledge_graph("Raajan is my friend from the club."))

    assert result == "Stored in the knowledge graph: 1 entities and 0 relationships created or updated."
    (text, prompt), = prompts_seen
    assert text == "Yadeesh (user): Raajan is my friend from the club."
    assert prompt.startswith(prompt_texts.KNOWLEDGE_GRAPH_EXTRACTION_PROMPT)  # the 7 types etc.
    assert prompt.endswith(prompt_texts.KNOWLEDGE_GRAPH_EXPLICIT_NOTE)
    assert memory.kg.learned_from.startswith("told directly by Yadeesh on ")


def test_relationship_names_come_from_one_fixed_list():
    from rag.knowledge_graph import RELATION_TYPES, canonical_relation

    assert canonical_relation("MEMBER_OF") == ("MEMBER_OF", False)
    assert canonical_relation("pursuing") == ("PREPARES_FOR", False)
    assert canonical_relation("STRONGLY_PREFERS") == ("PREFERS", False)
    # Synonyms written the other way round are stored reversed.
    assert canonical_relation("INVOLVES") == ("PART_OF", True)
    assert canonical_relation("HAS_MEMBER") == ("MEMBER_OF", True)
    assert canonical_relation("MANAGES") == ("REPORTS_TO", True)
    # The entity types settle what a vague name means.
    assert canonical_relation("WORKS_WITH", "Person", "Tool") == ("USES", False)
    assert canonical_relation("WORKS_WITH", "Person", "Person") == ("WORKS_WITH", False)
    assert canonical_relation("BELONGS_TO", "Project", "Organization") == ("PART_OF", False)
    # Never a new name: unknown names are read by their words, else a plain link.
    assert canonical_relation("COMPLETED_MORE_THAN_200_PROBLEMS_ON") == ("WORKED_ON", False)
    assert canonical_relation("SOMETHING_ODD") == ("RELATED_TO", False)
    assert all(canonical_relation(name)[0] == name for name in RELATION_TYPES)


def test_extraction_output_is_normalized_before_validation():
    from rag.knowledge_graph import normalize_candidates

    out = normalize_candidates({"candidates": {
        "entities": [{"id": "Docker", "type": "Technology"}, {"id": "Vision Research", "type": "Research"}],
        "relationships": [
            {"source": "Yadeesh", "target": "Docker", "relation_type": "works with"},
            {"source": "Vision Research", "target": "ViT", "relation_type": "INVOLVES"},
        ],
    }})
    assert [e["type"] for e in out["candidates"]["entities"]] == ["Tool", "Project"]
    assert out["candidates"]["relationships"] == [
        {"source": "Yadeesh", "target": "Docker", "relation_type": "USES"},
        {"source": "ViT", "target": "Vision Research", "relation_type": "PART_OF"},
    ]


def test_the_validator_sees_every_field_and_the_relationship_list(tmp_path, monkeypatch):
    import pandas

    import core.llm

    from rag.knowledge_graph import KnowledgeGraph

    sent = []

    class FakeLLM:
        def invoke(self, prompt):
            sent.append(prompt)

            class Reply:
                content = '{"resolution": {"entities": [], "relationships": []}}'

            return Reply()

    monkeypatch.setattr(core.llm, "build_llm", lambda *a, **k: FakeLLM())
    kg = KnowledgeGraph(db_path=str(tmp_path / "kg" / "graph.db"))
    long_description = "Campus placement preparation for the 2027 batch at VIT Chennai, " * 3
    lookalikes = pandas.DataFrame([{
        "id": "Campus placements", "type": "Concept", "description": long_description,
        "updated_at": "2026-10-04T23:09:20", "relations": ["PREPARES_FOR with Yadeesh"], "similarity_score": 0.94,
    }])
    kg.validate_entity_relation(lookalikes, {"candidates": {"entities": [], "relationships": []}})
    kg.generate_entity_relation("Yadeesh (user): hello")

    validation, extraction = sent
    assert long_description.strip() in validation  # not "id ... similarity_score"
    assert "PREPARES_FOR with Yadeesh" in validation and "2026-10-04T23:09:20" in validation
    for prompt in (validation, extraction):
        assert "{relation_types}" not in prompt
        assert "- REPORTS_TO: a person reports to their manager" in prompt
    kg.close()


def test_lookalikes_are_only_the_closest_few(tmp_path):
    from rag.knowledge_graph import LOOKALIKE_TOP_K, KnowledgeGraph

    kg = KnowledgeGraph(db_path=str(tmp_path / "kg" / "graph.db"))
    same = (np.ones(384) / np.sqrt(384)).tolist()
    kg._compute_embedding = lambda text, is_query=False: same  # every entity looks alike
    for i in range(12):
        kg.add_entity(f"Tool {i}", "Tool", "", f"tool number {i}")

    found = kg.search_similar_node([{"id": "tool 3", "type": "Tool", "description": "x", "search_keywords": []}])
    assert len(found) <= LOOKALIKE_TOP_K + 1  # the closest few, plus the exact-name match
    assert "Tool 3" in set(found["id"])
    kg.close()


def test_unknown_relationship_names_are_mapped_by_meaning():
    from rag.knowledge_graph import canonical_relation

    # Names the model actually wrote into the real graph.
    assert canonical_relation("USES_TOOL", "Project", "Tool") == ("USES", False)
    assert canonical_relation("CAREER_GOAL", "Person", "Concept") == ("PREPARES_FOR", False)
    assert canonical_relation("TARGETS_ROLE", "Person", "Concept") == ("PREPARES_FOR", False)
    assert canonical_relation("CONSIDERS_ROLE", "Person", "Concept") == ("INTERESTED_IN", False)
    assert canonical_relation("RESEARCHED", "Person", "Project") == ("WORKED_ON", False)
    assert canonical_relation("INVOLVED_WITH", "Person", "Organization") == ("MEMBER_OF", False)
    assert canonical_relation("COMPLETED_MORE_THAN_200_PROBLEMS_ON", "Person", "Tool") == ("USES", False)
    # Names never seen before: their words decide (whole words only).
    assert canonical_relation("ASPIRES_TO_BECOME", "Person", "Concept") == ("PREPARES_FOR", False)
    assert canonical_relation("FOCUSES_HEAVILY_ON", "Person", "Project") == ("WORKS_ON", False)  # not USES
    assert canonical_relation("ENJOYS", "Person", "Concept") == ("PREFERS", False)
    assert canonical_relation("RANDOM_LINK", "Person", "Concept") == ("RELATED_TO", False)


def test_names_differing_only_in_punctuation_are_one_entity(tmp_path):
    kg = _small_graph(tmp_path)
    kg.add_entity("College-Club", "Organization", "", "Meets on Fridays")
    assert kg.count_entities() == 4
    assert "Meets on Fridays" in kg._stored_entity("College Club")["description"]
    kg.add_relationship("Ann", "college-club", "member of")
    assert ("Ann", "MEMBER_OF", "College Club") in _relations(kg)
    kg.close()
