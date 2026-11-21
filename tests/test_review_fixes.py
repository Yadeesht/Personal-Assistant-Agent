"""Regression tests for the fixes from the full-repository bug review
(docs/BUG_REVIEW.md). No Google API or model calls are made: services and models
are faked."""

import asyncio
import json
import os
import sys
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import MagicMock

import aiosqlite
import pytest

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

import app_tools.tools.rag_tools as rag_tools  # noqa: E402
import core.agent as agent  # noqa: E402
import utils.memory_manager as memory_manager  # noqa: E402


@pytest.fixture
def quiet_memory(tmp_path, monkeypatch):
    """Agent nodes log to a temp DB and recall nothing."""
    db_path = tmp_path / "memory.db"
    monkeypatch.setattr(memory_manager, "MEMORY_DB", db_path)
    monkeypatch.setattr(agent, "MEMORY_DB", db_path)
    monkeypatch.setattr(rag_tools, "get_profile", lambda: ("", set()))
    monkeypatch.setattr(rag_tools, "recall_memory", lambda *a, **k: "")
    return db_path


# --- main.py: a message starting with a command word is still a message -------------


def test_bare_command_words_only_count_on_their_own():
    from main import parse_command

    assert parse_command("new meeting with Bob at 5") == ("", "")
    assert parse_command("help me write an email") == ("", "")
    assert parse_command("clear my calendar on Friday") == ("", "")
    assert parse_command("thread summary please") == ("", "")
    assert parse_command("new") == ("new", "")
    assert parse_command("  EXIT ") == ("exit", "")
    assert parse_command("/thread 12") == ("/thread", "12")
    assert parse_command("/new") == ("/new", "")
    assert parse_command("/bogus") == ("/bogus", "")


# --- core/agent.py ------------------------------------------------------------------


def test_route_to_agent_rejects_an_unknown_agent():
    result = agent.route_to_agent.func(
        state={"messages": [HumanMessage(content="hi")]},
        tool_call_id="call_1",
        agent="email_agent",
    )
    update = result.update
    assert update["current_agent"] == "supervisor"
    assert "unknown agent" in update["supervisor_messages"][0].content
    assert "communication_messages" not in update


def test_worker_network_error_is_shown_in_the_workers_own_list(quiet_memory):
    import httpx

    class FailingLLM:
        async def ainvoke(self, messages):
            raise httpx.ConnectError("offline")

    node = agent.agent_node_factory(FailingLLM(), "p {current_time}", "planning_agent")
    state = {
        "messages": [HumanMessage(content="What is on my calendar?", id="m1")],
        "planning_messages": [HumanMessage(content="[Handoff]", name="supervisor")],
    }
    update = asyncio.run(node(state, {"configurable": {"thread_id": "1"}}))
    # main.py shows the active agent's own list; the error used to land only in "messages".
    assert "Network unavailable" in update["planning_messages"][-1].content
    assert update["current_agent"] == "planning_agent"


def _code_state(reply: str) -> dict:
    approval = AIMessage(
        content="I have generated the following Python code:\n\n```python\nprint('hi')\n```\n\nreply approve",
        name="code_agent",
    )
    return {
        "messages": [HumanMessage(content=reply, id="u2")],
        "code_messages": [
            HumanMessage(content="[Handoff from supervisor] ...", name="supervisor"),
            approval,
            HumanMessage(content=reply),
        ],
        "current_agent": "code_agent",
    }


def test_declining_generated_code_hands_back_without_running(quiet_memory, monkeypatch):
    ran = []
    monkeypatch.setattr(
        agent.CodeExecutionAgent, "_execute_in_sandbox", lambda *a, **k: ran.append(a)
    )
    node = agent.code_execution_factory(MagicMock(), {}, "code_agent")
    update = asyncio.run(node(_code_state("No"), {}))
    assert not ran
    assert update["current_agent"] == "supervisor"
    handoff = update["supervisor_messages"][-1]
    assert "declined" in handoff.content and handoff.name == "code_agent"


def test_approved_code_runs_and_the_result_reaches_the_supervisor(quiet_memory, monkeypatch):
    from core.state import TaskSpec

    ran = []

    async def fake_intent(self, messages):
        return TaskSpec(primary_goal="say hi", required_tools_hint=[], context_variables={})

    async def fake_sandbox(self, code, tool_map=None):
        ran.append(code)
        return {"status": "success", "summary": "printed hi", "full_output": {"summary": "printed hi"}}

    monkeypatch.setattr(agent.CodeExecutionAgent, "_resolve_intent", fake_intent)
    monkeypatch.setattr(agent.CodeExecutionAgent, "_execute_in_sandbox", fake_sandbox)
    node = agent.code_execution_factory(MagicMock(), {}, "code_agent")
    update = asyncio.run(node(_code_state("yes"), {}))

    assert ran == ["print('hi')"]
    # The supervisor answers next, so it must see the result in its own list.
    assert "printed hi" in update["supervisor_messages"][-1].content
    assert update["current_agent"] == "supervisor"


# --- utils/helper.py ----------------------------------------------------------------


def test_token_counting_survives_a_missing_tokenizer(monkeypatch):
    import utils.helper as helper

    def offline(*a, **k):
        raise OSError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr(helper, "_encoding", None)
    monkeypatch.setattr(helper.tiktoken, "encoding_for_model", offline)
    monkeypatch.setattr(helper.tiktoken, "get_encoding", offline)
    assert helper.count_tokens("x" * 40) == 10
    assert helper.count_tokens([HumanMessage(content="x" * 40)]) == 16


# --- Gmail --------------------------------------------------------------------------


def test_html_only_email_body_is_read_as_text():
    from app_tools.tools.google.gmail_tools import extract_email_body

    msg = EmailMessage()
    msg.set_content("<html><body><p>Your invoice is <b>ready</b></p></body></html>", subtype="html")
    msg.add_attachment(b"%PDF", maintype="application", subtype="pdf", filename="a.pdf")
    body = extract_email_body(msg)
    assert "invoice" in body and "<b>" not in body


def test_unread_emails_with_bad_parameters_returns_an_error():
    from app_tools.tools.google import gmail_tools

    result = asyncio.run(gmail_tools.get_unread_emails(date=-5))
    assert result["count"] == 0 and "Invalid parameters" in result["error"]


# --- Calendar -----------------------------------------------------------------------


def test_custom_reminders_are_sent_to_the_calendar(monkeypatch):
    from app_tools.tools.google import calendar_tools

    service = MagicMock()
    service.events.return_value.insert.return_value.execute.return_value = {
        "htmlLink": "https://cal/x",
        "summary": "Standup",
    }
    monkeypatch.setattr(calendar_tools, "get_service", lambda: service)
    result = json.loads(
        asyncio.run(
            calendar_tools.create_event(
                summary="Standup",
                start_time="2026-03-01T09:00:00",
                end_time="2026-03-01T09:30:00",
                reminders=[{"method": "popup", "minutes": 15}],
            )
        )
    )
    assert result["status"] == "success"
    body = service.events.return_value.insert.call_args.kwargs["body"]
    # Before the fix the overrides were dropped and the event had no reminders at all.
    assert body["reminders"] == {
        "useDefault": False,
        "overrides": [{"method": "popup", "minutes": 15}],
    }


def test_a_free_time_range_is_not_reported_as_an_error(monkeypatch):
    from app_tools.tools.google import calendar_tools

    service = MagicMock()
    service.events.return_value.list.return_value.execute.return_value = {"items": []}
    monkeypatch.setattr(calendar_tools, "get_service", lambda: service)
    result = json.loads(asyncio.run(calendar_tools.get_events(time_min="2026-03-01")))
    assert result["status"] == "success" and "No events" in result["message"]


def test_times_without_seconds_get_the_ist_offset():
    from app_tools.tools.google.calendar_tools import _correct_time_format_for_api

    assert _correct_time_format_for_api("2026-03-01T09:30", "t") == "2026-03-01T09:30:00+05:30"


# --- Drive --------------------------------------------------------------------------


def test_create_drive_file_with_content_uploads_one_file(monkeypatch):
    from app_tools.tools.google import gdrive_tools

    service = MagicMock()
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "root",
        "mimeType": gdrive_tools.FOLDER_MIME_TYPE,
    }
    service.files.return_value.create.return_value.execute.return_value = {
        "id": "f1",
        "name": "notes.txt",
        "webViewLink": "https://drive/f1",
    }
    monkeypatch.setattr(gdrive_tools, "get_service", lambda: service)
    result = json.loads(
        asyncio.run(gdrive_tools.create_drive_file(file_name="notes.txt", content="hello"))
    )
    # Before the fix this always failed with an undefined `parsed_url`.
    assert result["status"] == "success", result
    assert service.files.return_value.create.call_count == 1


# --- Search -------------------------------------------------------------------------


def test_site_restricted_search_runs(monkeypatch):
    from app_tools.tools.google import gsearch_tools

    service = MagicMock()
    service.cse.return_value.list.return_value.execute.return_value = {
        "searchInformation": {"totalResults": "1", "searchTime": 0.1},
        "items": [{"title": "T", "link": "https://x", "snippet": "s"}],
    }
    monkeypatch.setattr(gsearch_tools, "get_service", lambda: service)
    monkeypatch.setenv("GOOGLE_PSE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_PSE_ENGINE_ID", "cx")
    text = asyncio.run(gsearch_tools.search_custom_siterestrict(q="llm", sites=["a.com"]))
    assert "https://x" in text
    assert service.cse.return_value.list.call_args.kwargs["q"] == "llm (site:a.com)"


# --- Forms --------------------------------------------------------------------------


def test_form_description_is_set_after_creation(monkeypatch):
    from app_tools.tools.google import gform_tools

    service = MagicMock()
    service.forms.return_value.create.return_value.execute.return_value = {
        "formId": "F1",
        "info": {"title": "Survey"},
    }
    monkeypatch.setattr(gform_tools, "get_service", lambda: service)
    result = json.loads(
        asyncio.run(gform_tools.create_form(title="Survey", description="Tell us"))
    )
    assert result["status"] == "success"
    # forms.create rejects a description; it goes in a batchUpdate.
    assert "description" not in service.forms.return_value.create.call_args.kwargs["body"]["info"]
    update = service.forms.return_value.batchUpdate.call_args.kwargs["body"]["requests"][0]
    assert update["updateFormInfo"]["info"] == {"description": "Tell us"}


def test_publish_settings_use_the_api_fields(monkeypatch):
    from app_tools.tools.google import gform_tools

    service = MagicMock()
    monkeypatch.setattr(gform_tools, "get_service", lambda: service)
    result = json.loads(
        asyncio.run(gform_tools.set_publish_settings(form_id="F1", is_accepting_responses=False))
    )
    assert result["status"] == "success"
    body = service.forms.return_value.setPublishSettings.call_args.kwargs["body"]
    assert body == {
        "publishSettings": {
            "publishState": {"isPublished": True, "isAcceptingResponses": False}
        },
        "updateMask": "publishState",
    }


# --- Docs ---------------------------------------------------------------------------


def test_doc_tools_report_invalid_parameters_instead_of_crashing():
    from app_tools.tools.google import gdocs_tools

    inspect = asyncio.run(gdocs_tools.inspect_doc_structure(document_id=""))
    assert inspect["status"] == "error"
    insert = asyncio.run(gdocs_tools.insert_doc_elements("doc", "table", index=-1))
    assert insert["status"] == "error"
    table = asyncio.run(gdocs_tools.create_table_with_data("doc", [["a"], ["b", "c"]], 1))
    assert table["status"] == "error"


def test_inserted_text_can_be_formatted_without_end_index(monkeypatch):
    from app_tools.tools.google import gdocs_tools

    service = MagicMock()
    monkeypatch.setattr(gdocs_tools, "get_service", lambda: service)
    result = asyncio.run(
        gdocs_tools.modify_doc_text(
            document_id="d" * 30, start_index=5, text="Title", bold=True
        )
    )
    assert result["status"] == "success", result
    requests = service.documents.return_value.batchUpdate.call_args.kwargs["body"]["requests"]
    assert requests[1]["updateTextStyle"]["range"] == {"startIndex": 5, "endIndex": 10}


def _doc_with_tables(*starts):
    def table(start):
        cell = {
            "startIndex": start + 1,
            "endIndex": start + 3,
            "content": [{"paragraph": {"elements": [{"startIndex": start + 2, "textRun": {"content": "\n"}}]}}],
        }
        return {"startIndex": start, "endIndex": start + 4, "table": {"tableRows": [{"tableCells": [cell]}]}}

    return {"body": {"content": [table(s) for s in starts]}}


def test_new_table_is_filled_even_when_it_is_not_the_last_one():
    from app_tools.helper.docs_managers import TableOperationManager

    docs = [_doc_with_tables(100), _doc_with_tables(10, 104)]  # before / after the insert
    service = MagicMock()
    service.documents.return_value.get.return_value.execute.side_effect = lambda: docs[
        min(service.documents.return_value.get.call_count - 1, 1)
    ]
    manager = TableOperationManager(service)
    ok, _, meta = asyncio.run(manager.create_and_populate_table("doc", [["x"]], 10, False))
    assert ok and meta["table_index"] == 0
    insert = service.documents.return_value.batchUpdate.call_args_list[-1].kwargs["body"]
    # The new table (at 10) gets the text, not the existing one that now starts at 104.
    assert insert["requests"][0]["insertText"]["location"]["index"] == 12


def test_first_page_header_never_overwrites_the_default_one():
    from app_tools.helper.docs_managers import HeaderFooterManager

    doc = {
        "documentStyle": {"defaultHeaderId": "kix.default"},
        "headers": {"kix.default": {"headerId": "kix.default", "content": []}},
    }
    manager = HeaderFooterManager(MagicMock())
    found = asyncio.run(manager._find_target_section(doc, "header", "FIRST_PAGE_ONLY"))
    assert found == (None, None)
    found = asyncio.run(manager._find_target_section(doc, "header", "DEFAULT"))
    assert found[1] == "kix.default"


def test_batch_update_doc_accepts_find_replace(monkeypatch):
    from app_tools.tools.google import gdocs_tools

    service = MagicMock()
    service.documents.return_value.batchUpdate.return_value.execute.return_value = {"replies": [{}]}
    monkeypatch.setattr(gdocs_tools, "get_service", lambda: service)
    result = asyncio.run(
        gdocs_tools.batch_update_doc(
            "doc", [{"type": "find_replace", "find_text": "old", "replace_text": "new"}]
        )
    )
    assert result["status"] == "success", result
    request = service.documents.return_value.batchUpdate.call_args.kwargs["body"]["requests"][0]
    assert request["replaceAllText"]["containsText"]["text"] == "old"


# --- Knowledge graph ----------------------------------------------------------------


def test_deleting_an_entity_with_a_quote_in_its_name(tmp_path, monkeypatch):
    from rag.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=tmp_path / "kg" / "graph.db")
    monkeypatch.setattr(kg, "_compute_embedding", lambda text, is_query=False: [0.1] * 384)
    kg.add_entity("O'Brien", "Person", "friend", "A friend")
    assert kg.count_entities() == 1
    kg.delete_entity("O'Brien")
    assert kg.count_entities() == 0


# --- Security choices confirmed with the user ----------------------------------------


def test_generated_code_cannot_call_confirm_gated_tools():
    from langchain_core.tools import tool

    from core.codeagent import CodeExecutionAgent

    @tool
    def send_email(recipient_id: str) -> str:
        """Send."""
        return "sent"

    @tool
    def search_emails(query: str) -> str:
        """Search."""
        return "[]"

    code_agent = CodeExecutionAgent(MagicMock(), {"communication": [send_email, search_emails]})
    assert set(code_agent._create_tool_map(["communication"])) == {"search_emails"}
    schemas = asyncio.run(code_agent._load_tool_schemas(["communication"]))
    assert [t["name"] for t in schemas["communication"]] == ["search_emails"]


def test_drive_upload_refuses_credentials(monkeypatch):
    from app_tools.tools.google import gdrive_tools

    service = MagicMock()
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "root",
        "mimeType": gdrive_tools.FOLDER_MIME_TYPE,
    }
    monkeypatch.setattr(gdrive_tools, "get_service", lambda: service)
    token = gdrive_tools._PROJECT_DIR / "app_tools" / "cred" / "gmail_token.json"
    result = json.loads(
        asyncio.run(gdrive_tools.create_drive_file(file_name="x", fileUrl=token.as_uri()))
    )
    assert result["status"] == "error" and "Refusing" in result["error"]
    assert not service.files.return_value.create.called
