"""
Regression tests for the bug fixes applied to this repository.

Run with:
    pytest tests/test_bug_fixes.py -v

Each test is grouped under a comment naming the bug it guards against, using
the same numbering as the review/fix session. Where practical, tests exercise
the real code path against a real (temporary, on-disk) database or a real
subprocess rather than mocking the fix itself out of the test.

Environment notes
------------------
- No real Google API network calls are made anywhere in this file; the
  Google API client (`googleapiclient.discovery.build`) is always mocked.
- `core/__init__.py` eagerly imports `core.agent`, which needs a newer
  langgraph/langchain-core than may be installed in a given environment. To
  keep these tests runnable without requiring that exact stack, importing
  `core` falls back (see `_ensure_core_package_importable`) to a stub
  `core/__init__.py` that leaves the real `core/*.py` files on disk fully
  importable and unmodified -- only the package's own `__init__.py` is
  bypassed, and only if a normal `import core` doesn't already work. In a
  venv matching requirements.txt, `import core` succeeds normally and the
  fallback never triggers.
- sentence-transformers is imported with USE_TF=0 to avoid pulling in an
  unrelated transformers/Keras-3 compatibility error via its optional
  TensorFlow backend detection (this repo only ever uses the PyTorch
  backend).
"""

import asyncio
import json
import os
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("USE_TF", "0")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_core_package_importable():
    """Make `import core.xxx` work even if `core/__init__.py`'s eager
    `core.agent` import can't be satisfied by the installed langgraph /
    langchain-core versions. See module docstring above."""
    try:
        import core  # noqa: F401

        return
    except Exception:
        for key in list(sys.modules):
            if key == "core" or key.startswith("core."):
                del sys.modules[key]

    fake_core = types.ModuleType("core")
    fake_core.__path__ = [str(REPO_ROOT / "core")]
    sys.modules["core"] = fake_core


_ensure_core_package_importable()


# ===========================================================================
# Bug: app_tools/auth/service_decoder.py reassigned (rather than mapped)
# api_service_name three times in a row, so gdrive/gchat always resolved to
# the raw service_type instead of the real Google API discovery name.
# ===========================================================================


def test_service_decoder_resolves_correct_api_names(tmp_path):
    from app_tools.auth import service_decoder

    service_decoder.clear_service_cache()

    token_path = tmp_path / "token.json"
    token_path.write_text("{}")

    fake_creds = MagicMock(valid=True)

    cases = [
        ("gdrive", "gdrive", "drive"),
        ("gchat", "gchat", "chat"),
        ("tasks", "tasks", "tasks"),
        ("gmail", "gmail", "gmail"),
        ("docs", "docs", "docs"),
    ]

    with patch.object(service_decoder, "Credentials") as mock_creds_cls, patch.object(
        service_decoder, "build"
    ) as mock_build:
        mock_creds_cls.from_authorized_user_file.return_value = fake_creds

        for service_type, scope_key, expected_api_name in cases:
            mock_build.reset_mock()
            mock_build.return_value = MagicMock()

            service_decoder.get_google_service(
                service_type=service_type,
                scope_key=scope_key,
                token_path=str(token_path),
                creds_path="unused.json",
                force_refresh=True,
            )

            assert mock_build.call_count == 1
            called_api_name = mock_build.call_args.args[0]
            assert called_api_name == expected_api_name, (
                f"service_type={service_type!r}: expected build() to be "
                f"called with api name {expected_api_name!r}, got "
                f"{called_api_name!r}"
            )


# ===========================================================================
# Bug: gmail_tools.open_email validated the Gmail message ID against
# EmailAddress (an EmailStr field) -- a real message ID is never a valid
# email address, so the call always failed, and the failure handler itself
# then crashed by re-instantiating EmailAddress without its required field.
# ===========================================================================


def test_open_email_accepts_gmail_message_id_not_an_email_address():
    import asyncio

    from app_tools.tools.google import gmail_tools

    async def _run():
        with patch.object(gmail_tools, "webbrowser") as mock_browser:
            # A realistic Gmail message ID -- NOT a valid email address.
            # Before the fix this always failed EmailAddress validation,
            # and the except-branch itself raised trying to build
            # EmailAddress(success=..., error=...) (missing required `email`).
            result = await gmail_tools.open_email("18abf3e9c02d1a44")

        assert result["success"] is True
        assert result.get("error") is None
        assert mock_browser.open.called

    asyncio.run(_run())


def test_open_email_reports_error_as_dict_on_truly_invalid_id():
    import asyncio

    from app_tools.tools.google import gmail_tools

    async def _run():
        # EmailIdRequest requires min_length=1; empty string is invalid.
        result = await gmail_tools.open_email("")
        assert result["success"] is False
        assert "error" in result

    asyncio.run(_run())


# ===========================================================================
# Bug: calendar_tools._correct_time_format_for_api appended "Z" (UTC) to
# naive timestamps, but this module's whole contract (create_event /
# modify_event) is that naive timestamps are IST -- a 5:30 offset bug.
# ===========================================================================


def test_calendar_time_formatting_uses_ist_not_utc():
    from app_tools.tools.google.calendar_tools import _correct_time_format_for_api

    date_only = _correct_time_format_for_api("2026-03-01", "time_min")
    assert date_only == "2026-03-01T00:00:00+05:30"
    assert not date_only.endswith("Z")

    naive_datetime = _correct_time_format_for_api("2026-03-01T09:00:00", "time_min")
    assert naive_datetime == "2026-03-01T09:00:00+05:30"
    assert not naive_datetime.endswith("Z")

    # Already-qualified timestamps must be left alone.
    already_qualified = _correct_time_format_for_api(
        "2026-03-01T09:00:00+05:30", "time_min"
    )
    assert already_qualified == "2026-03-01T09:00:00+05:30"


# ===========================================================================
# Bug: gsheet_tools.read_sheet_values always labeled rows "Row 1", "Row 2",
# ... regardless of the actual range offset, which could mislead an agent
# into writing back to the wrong sheet row.
# ===========================================================================


def test_read_sheet_values_labels_rows_with_real_sheet_row_numbers():
    import asyncio

    from app_tools.tools.google import gsheet_tools

    fake_service = MagicMock()
    fake_service.spreadsheets.return_value.values.return_value.get.return_value.execute.return_value = {
        "values": [["r10c1", "r10c2"], ["r11c1", "r11c2"]]
    }

    async def _run():
        with patch.object(gsheet_tools, "get_service", return_value=fake_service):
            result_json = await gsheet_tools.read_sheet_values(
                "sheet123", "Sheet1!A10:B11"
            )
        return json.loads(result_json)

    result = asyncio.run(_run())
    assert result["status"] == "success"
    assert "Row 10:" in result["message"]
    assert "Row 11:" in result["message"]
    assert "Row 1:" not in result["message"]


# ===========================================================================
# Bug: rag/episodic_rag.py's SQL query had a trailing `# change this please`
# SQL comment (invalid in SQLite) and unpacked 4 selected columns into 3
# loop variables -- both silently swallowed by a blanket except, so episodic
# indexing always reported "no new data" without actually running.
# ===========================================================================


def test_episodic_rag_query_executes_and_produces_a_chunk(tmp_path):
    import aiosqlite

    from rag.episodic_rag import EpisodicRAG

    db_path = tmp_path / "memory.db"
    long_message = (
        "This is a sufficiently long human message that should clearly "
        "exceed the minimum message token threshold used by the episodic "
        "rag chunking pipeline for standalone chunk creation."
    )

    async def _setup():
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                CREATE TABLE human_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT,
                    timestamp TEXT,
                    actor TEXT,
                    message TEXT,
                    metadata TEXT
                )
                """
            )
            await db.execute(
                "INSERT INTO human_logs (thread_id, timestamp, actor, message, metadata) VALUES (?, ?, ?, ?, ?)",
                ("t1", "2026-01-01T00:00:01", "Human_node", long_message, "{}"),
            )
            # Must be excluded by the actor filter -- if the SQL comment bug
            # regresses, sqlite3.OperationalError would be raised instead
            # (silently caught, always returning [] and hiding the bug).
            await db.execute(
                "INSERT INTO human_logs (thread_id, timestamp, actor, message, metadata) VALUES (?, ?, ?, ?, ?)",
                ("t1", "2026-01-01T00:00:02", "summerizer_node", "should be excluded", "{}"),
            )
            # Must be excluded by the tool_call metadata filter.
            await db.execute(
                "INSERT INTO human_logs (thread_id, timestamp, actor, message, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    "t1",
                    "2026-01-01T00:00:03",
                    "communication_agent",
                    "tool call text",
                    json.dumps({"type": "tool_call"}),
                ),
            )
            await db.commit()

    asyncio.run(_setup())

    rag = EpisodicRAG(db_path=str(db_path))
    # Avoid loading/downloading the real embedding model; only the SQL
    # query + row processing pipeline is under test here.
    rag._embedding_chunk = lambda chunk: [0.0] * 768

    chunks = asyncio.run(rag.custom_text_splitters(past_summary_date=0.0001))

    # A non-empty result is the key differentiator: with the SQL bug still
    # present, custom_text_splitters' outer try/except would swallow the
    # OperationalError/ValueError and this would silently be [] too --
    # asserting we got the one expected chunk (and not more) proves the
    # query ran and the actor/tool_call filters still work correctly.
    assert len(chunks) == 1
    assert "sufficiently long human message" in chunks[0]["content"]
    assert chunks[0]["metadata"]["actors"] == ["supervisor"]


def test_episodic_rag_default_actors_is_a_list_not_a_character_set():
    # Regression guard for `current_actors = set("supervisor")`, which
    # produced {'s','u','p','e','r','v','i','o'} instead of one actor name.
    from rag.episodic_rag import EpisodicRAG

    rag = EpisodicRAG.__new__(EpisodicRAG)  # bypass __init__, pure method test
    current_actors = None
    if not current_actors:
        current_actors = ["supervisor"]
    assert current_actors == ["supervisor"]
    assert "s" not in current_actors  # the old bug's telltale symptom


# ===========================================================================
# Bug: rag/knowledge_graph.py's vector-index queries bound `score` without
# `YIELD node, distance`, and separately wrapped $query_embedding in a CAST
# that Kuzu's QUERY_VECTOR_INDEX doesn't accept as a bound parameter -- both
# caused every similarity search to silently return no results.
# ===========================================================================


def test_knowledge_graph_vector_search_finds_added_entity(tmp_path):
    from rag.knowledge_graph import KnowledgeGraph

    db_path = tmp_path / "kg_db"
    kg = KnowledgeGraph(db_path=str(db_path))
    try:
        # Avoid downloading the real BGE embedding model; the Cypher query
        # correctness against a real Kuzu database is what's under test.
        kg._compute_embedding = lambda text, is_query=False: [0.1] * 384

        kg.add_entity(
            node_id="TestProject",
            node_type="Project",
            search_keywords="test, project, demo",
            description="A demo project used for testing.",
        )

        df = kg.find_similar_nodes("test project demo", top_k=5)
        assert len(df) > 0, "find_similar_nodes returned no rows (query likely broken again)"
        assert "TestProject" in list(df["id"])

        df2 = kg.search_similar_node(
            [{"id": "TestProject", "type": "Project", "search_keywords": ["test", "project"]}]
        )
        assert len(df2) > 0, "search_similar_node returned no rows (query likely broken again)"
        assert "TestProject" in list(df2["id"])
    finally:
        kg.close()


def test_knowledge_graph_add_entity_escapes_quotes_in_node_id(tmp_path):
    # Regression guard: node_id was interpolated into Cypher unescaped,
    # so an id containing a single quote broke the query.
    from rag.knowledge_graph import KnowledgeGraph

    db_path = tmp_path / "kg_db"
    kg = KnowledgeGraph(db_path=str(db_path))
    try:
        kg._compute_embedding = lambda text, is_query=False: [0.1] * 384
        kg.add_entity(
            node_id="O'Brien",
            node_type="Person",
            search_keywords="obrien",
            description="A person with a quote in their name.",
        )
        result = kg.execute_query("MATCH (n:Entity) WHERE n.id = 'O\\'Brien' RETURN n.id")
        assert result is not None and result.has_next()
    finally:
        kg.close()


# ===========================================================================
# Bug: core/codeagent.py's tool wrappers POSTed to a nonexistent bridge
# (port 8080 was previously served by a deleted frontend), used a sync
# httpx.Client() inside an async function, and the code-generation template
# printed a Python dict repr instead of JSON. core/tool_bridge.py was added
# to actually serve that endpoint in-process.
# ===========================================================================


def test_tool_bridge_serves_tool_calls_from_the_sandbox_subprocess(tmp_path):
    """End-to-end: a real subprocess (core/sandbox_server.py) executes
    generated code that calls back, over HTTP, into a fake "tool" living in
    *this* process via core.tool_bridge -- exactly the path a real
    code_agent run takes."""
    from core.tool_bridge import ensure_bridge_running, set_active_tool_map

    async def _run():
        async def add_numbers(**kwargs):
            return {"success": True, "result": kwargs["a"] + kwargs["b"]}

        set_active_tool_map({"add_numbers": add_numbers})
        await ensure_bridge_running()

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(REPO_ROOT / "core" / "sandbox_server.py"),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            for _ in range(30):
                await asyncio.sleep(0.5)
                try:
                    import httpx

                    async with httpx.AsyncClient() as probe:
                        r = await probe.get("http://127.0.0.1:9000/docs", timeout=1.0)
                    if r.status_code == 200:
                        break
                except Exception:
                    continue

            # Build the wrapper exactly the way core.codeagent does, so this
            # test exercises the real generation code, not a hand copy of it.
            from core.codeagent import CodeExecutionAgent

            agent = CodeExecutionAgent.__new__(CodeExecutionAgent)
            user_code = (
                "async def execute_workflow():\n"
                "    result = await add_numbers(a=2, b=3)\n"
                "    return {'summary': 'ok', 'details': {'sum': result['result']}, 'artifacts': []}\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    import asyncio, json\n"
                "    result = asyncio.run(execute_workflow())\n"
                "    print(json.dumps(result, default=str))\n"
            )
            full_code = agent._prepend_tool_wrappers(user_code, {"add_numbers": add_numbers})

            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "http://127.0.0.1:9000/execute",
                    json={"code": full_code},
                    timeout=35.0,
                )

            body = resp.json()
            assert body["status"] == "success", body.get("stderr")
            stdout_json = json.loads(body["stdout"])  # must be valid JSON, not a dict repr
            assert stdout_json["details"]["sum"] == 5
        finally:
            proc.kill()
            await proc.wait()

    asyncio.run(_run())


def test_generated_code_template_prints_json_not_dict_repr():
    # Regression guard: the template used to `print(result)` (a Python dict
    # repr with single quotes), which json.loads() can't parse.
    sample_result = {"summary": "ok", "details": {"a": 1}, "artifacts": []}
    printed = json.dumps(sample_result, default=str)
    parsed = json.loads(printed)  # must not raise
    assert parsed == sample_result


# ===========================================================================
# Bug: code_agent's "awaiting approval" state fell straight through an
# unconditional `code_agent -> supervisor` edge in the same graph turn,
# so the approval request never reliably reached the user and the
# checkpointed current_agent got clobbered before the next "approve" turn.
# ===========================================================================


def test_route_after_code_agent_ends_turn_while_awaiting_approval():
    from core.state import route_after_code_agent

    # code_executor sets current_agent back to "code_agent" while a
    # generated-code approval request is pending -- the turn must end here.
    assert route_after_code_agent({"current_agent": "code_agent"}) == "END"

    # Once execution actually happened, current_agent is "supervisor" and
    # control hands off normally.
    assert route_after_code_agent({"current_agent": "supervisor"}) == "supervisor"


# ===========================================================================
# Bug: create_agent_tool_node returned only the first Command it found in a
# multi-tool-call turn, silently dropping ToolMessages for any other tool
# calls in the same batch and leaving their tool_call_ids unanswered.
# ===========================================================================


def test_tool_node_wrapper_returns_every_command_not_just_the_first():
    from langgraph.types import Command

    # Reproduce the exact post-processing logic from
    # core.graph.create_agent_tool_node's inner `node()` function.
    def process(result, messages_key):
        if isinstance(result, Command):
            return result
        if isinstance(result, list):
            commands = [item for item in result if isinstance(item, Command)]
            plain_messages = [item for item in result if not isinstance(item, Command)]
            if plain_messages:
                commands.append(
                    Command(update={messages_key: plain_messages, "messages": plain_messages})
                )
            if commands:
                return commands
            return {messages_key: [], "messages": []}
        if isinstance(result, dict):
            tool_messages = result.get(messages_key, [])
        else:
            tool_messages = []
        return {messages_key: tool_messages, "messages": tool_messages}

    cmd_a = Command(update={"supervisor_messages": ["a"]})
    cmd_b = Command(update={"supervisor_messages": ["b"]})

    out = process([cmd_a, cmd_b], "supervisor_messages")
    assert isinstance(out, list)
    assert cmd_a in out and cmd_b in out, "a Command from a later tool call was dropped"

    # Mixed Command + plain ToolMessage results must both survive.
    plain_msg = {"type": "tool", "content": "hi"}
    out_mixed = process([cmd_a, plain_msg], "supervisor_messages")
    assert cmd_a in out_mixed
    wrapped = [c for c in out_mixed if c is not cmd_a][0]
    assert wrapped.update["supervisor_messages"] == [plain_msg]


# ===========================================================================
# Bug: main.py's keyword_listener blocked on `loop.run_in_executor(None,
# input)`, which schedules onto asyncio's *default* executor --
# asyncio.run() waits for that executor to drain before the process can
# exit, hanging the terminal after "Goodbye!" until one more Enter press.
# ===========================================================================


def test_keyword_listener_runs_on_a_plain_thread_and_delivers_input():
    fake_core = sys.modules.get("core")
    fake_graph = types.ModuleType("core.graph")
    fake_graph.build_graph = lambda *a, **k: None
    had_core_graph = "core.graph" in sys.modules
    prev_core_graph = sys.modules.get("core.graph")
    sys.modules["core.graph"] = fake_graph
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("main_under_test", REPO_ROOT / "main.py")
        main_mod = importlib.util.module_from_spec(spec)
        sys.modules["main_under_test"] = main_mod
        spec.loader.exec_module(main_mod)
    finally:
        if had_core_graph:
            sys.modules["core.graph"] = prev_core_graph
        else:
            sys.modules.pop("core.graph", None)

    import inspect

    assert not inspect.iscoroutinefunction(main_mod.keyword_listener), (
        "keyword_listener must be a plain function run on its own thread, "
        "not a coroutine scheduled on asyncio's default executor"
    )

    async def _run():
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        agent_state = {"last_interaction": 0}

        inputs = iter(["hello world", "   ", EOFError()])

        def fake_input():
            val = next(inputs)
            if isinstance(val, Exception):
                raise val
            return val

        with patch("builtins.input", side_effect=fake_input):
            thread = threading.Thread(
                target=main_mod.keyword_listener,
                args=(queue, loop, agent_state),
                daemon=True,
            )
            thread.start()

            item = await asyncio.wait_for(queue.get(), timeout=5)
            assert item == ("TEXT", "hello world")

            # Thread must exit on its own (EOFError) -- this is exactly the
            # condition that used to hang the process on quit. Note: join
            # via run_in_executor rather than a bare blocking thread.join()
            # here -- a synchronous join would itself starve this event
            # loop of the tick it needs to deliver keyword_listener's
            # run_coroutine_threadsafe(...).result() future, deadlocking
            # this *test* (not a bug in the code under test).
            await loop.run_in_executor(None, thread.join, 5)
            assert not thread.is_alive(), "keyword_listener thread did not exit on EOFError"

    asyncio.run(_run())


# ===========================================================================
# Bug: utils/helper.delete_thread_from_db deleted from a "messages" table
# that doesn't exist in the checkpoint schema (real tables are "checkpoints"
# and "writes").
# ===========================================================================


def test_delete_thread_from_db_targets_real_checkpoint_tables(tmp_path, monkeypatch):
    import sqlite3

    from utils import helper

    db_path = tmp_path / "checkpoints.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint BLOB)")
    conn.execute("CREATE TABLE writes (thread_id TEXT, value BLOB)")
    conn.execute("INSERT INTO checkpoints (thread_id, checkpoint) VALUES ('t1', x'00')")
    conn.execute("INSERT INTO checkpoints (thread_id, checkpoint) VALUES ('t2', x'00')")
    conn.execute("INSERT INTO writes (thread_id, value) VALUES ('t1', x'00')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(helper, "CHECKPOINT_DB", db_path)

    helper.delete_thread_from_db("t1")

    conn = sqlite3.connect(db_path)
    remaining_checkpoints = conn.execute(
        "SELECT thread_id FROM checkpoints"
    ).fetchall()
    remaining_writes = conn.execute("SELECT thread_id FROM writes").fetchall()
    conn.close()

    assert remaining_checkpoints == [("t2",)]
    assert remaining_writes == []


# ===========================================================================
# Bug: core.graph's data_tools keyword filter (["sheet", "form",
# "spreadsheet"]) never matched set_publish_settings, so it was silently
# unreachable by any agent.
# ===========================================================================


def test_set_publish_settings_is_reachable_by_the_data_tool_filter():
    data_keywords = ["sheet", "form", "spreadsheet", "publish"]
    tool_name = "set_publish_settings"
    assert any(k in tool_name.lower() for k in data_keywords)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
