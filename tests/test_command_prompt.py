"""
Tests for the terminal input line (utils/command_prompt.py) and prompt reloading.

Run with:
    pytest tests/test_command_prompt.py -v

The input line is driven with simulated keystrokes (prompt_toolkit's pipe input), so
the pop-up completion and highlighting are exercised without a real terminal.
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt_toolkit.document import Document  # noqa: E402
from prompt_toolkit.input import create_pipe_input  # noqa: E402
from prompt_toolkit.output import DummyOutput  # noqa: E402

from utils.command_prompt import (  # noqa: E402
    COMMANDS,
    SlashCommandCompleter,
    SlashCommandLexer,
    make_session,
)


def _completions(text):
    return [
        (c.text, c.display_meta_text)
        for c in SlashCommandCompleter().get_completions(Document(text), None)
    ]


def test_typing_a_slash_lists_every_command_with_what_it_does():
    assert _completions("/") == list(COMMANDS.items())
    assert dict(_completions("/"))["/new"] == "Save this chat to memory and start a fresh one"


def test_the_list_narrows_as_you_type_and_hides_otherwise():
    assert [text for text, _ in _completions("/th")] == ["/thread"]
    assert [text for text, _ in _completions("/RE")] == ["/reload"]
    assert _completions("/thread 12") == []  # arguments: no pop-up
    assert _completions("email Ravi") == []  # a normal message: no pop-up
    assert _completions("/nope") == []


def test_commands_are_highlighted_while_typed():
    def fragments(text):
        return SlashCommandLexer().lex_document(Document(text))(0)

    assert fragments("/thread 12") == [("class:command", "/thread"), ("", " 12")]
    assert fragments("/nope") == [("class:command.unknown", "/nope"), ("", "")]
    assert fragments("hello /new") == [("", "hello /new")]


def test_tab_completes_a_command_in_the_real_input_line():
    import threading
    import time

    with create_pipe_input() as keyboard:
        session = make_session(input=keyboard, output=DummyOutput())
        result = {}
        reader = threading.Thread(target=lambda: result.update(line=session.prompt("You ❯ ")))
        reader.start()
        # Typed like a person: the pop-up is filled in between key presses.
        for keys in ["/ne", "\t", "\r"]:
            time.sleep(0.5)
            keyboard.send_text(keys)
        reader.join(10)
        assert result.get("line") == "/new"


def test_a_failed_prompt_reload_keeps_the_running_graph(monkeypatch):
    # Load main.py the way the other tests do, with a stand-in graph module.
    fake_graph = types.ModuleType("core.graph")

    def broken_build_graph(*args, **kwargs):
        raise ImportError("cannot import name 'X' from 'config.settings'")

    fake_graph.build_graph = broken_build_graph
    monkeypatch.setitem(sys.modules, "core.graph", fake_graph)
    import importlib.util

    spec = importlib.util.spec_from_file_location("main_reload_test", REPO_ROOT / "main.py")
    main_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_mod)

    running_graph = object()
    graph, error = main_mod.try_reload_prompts({}, None, running_graph)
    assert graph is running_graph
    assert isinstance(error, ImportError)
