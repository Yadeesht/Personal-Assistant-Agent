"""
Tests for the model client options (core/llm.py).

Run with:
    pytest tests/test_llm.py -v

No requests are sent; only the client settings are checked.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

import core.llm as llm  # noqa: E402


@tool
def search_emails(query: str) -> str:
    """Search the user's Gmail."""
    return "[]"


@pytest.fixture
def azure(monkeypatch):
    monkeypatch.setattr(llm, "AZURE_AI_ENDPOINT", "https://res.cognitiveservices.azure.com/")
    monkeypatch.setattr(llm, "AZURE_AI_CREDENTIAL", "placeholder")


def test_luna_options_apply_temperature_everywhere_and_reasoning_only_with_tools(azure, monkeypatch):
    monkeypatch.setattr(llm, "MODEL_TEMPERATURE", 1.0)
    monkeypatch.setattr(llm, "MODEL_TOOL_REASONING_EFFORT", "none")

    plain = llm.build_llm("gpt-6-luna")
    assert plain.deployment_name == "gpt-6-luna"
    assert plain.temperature == 1.0
    assert plain.reasoning_effort is None  # summarizer / extraction keep the default reasoning

    with_tools = llm.build_llm_with_tools([search_emails], "gpt-6-luna", parallel_tool_calls=False)
    assert with_tools.bound.temperature == 1.0
    assert with_tools.bound.reasoning_effort == "none"
    assert with_tools.kwargs["parallel_tool_calls"] is False


def test_without_options_the_client_defaults_are_used(azure, monkeypatch):
    # e.g. gpt-4.1-mini: it rejects reasoning_effort, so nothing extra is sent.
    monkeypatch.setattr(llm, "MODEL_TEMPERATURE", None)
    monkeypatch.setattr(llm, "MODEL_TOOL_REASONING_EFFORT", None)

    with_tools = llm.build_llm_with_tools([search_emails], "gpt-4.1-mini")
    assert with_tools.bound.temperature == 0.7
    assert with_tools.bound.reasoning_effort is None


def test_a_full_request_url_as_endpoint_does_not_override_the_deployment(monkeypatch):
    # A URL copied from the portal used to pin every call to gpt-4.1-mini whatever
    # MODEL_NAME said (the deployment name ended up in the query string).
    monkeypatch.setattr(
        llm,
        "AZURE_AI_ENDPOINT",
        "https://res.cognitiveservices.azure.com/openai/deployments/gpt-4.1-mini/"
        "chat/completions?api-version=2025-01-01-preview",
    )
    monkeypatch.setattr(llm, "AZURE_AI_CREDENTIAL", "placeholder")

    client = llm.build_llm("gpt-6-luna")
    assert client.azure_endpoint == "https://res.cognitiveservices.azure.com/"
    assert str(client.root_client.base_url).endswith("/openai/deployments/gpt-6-luna/")
