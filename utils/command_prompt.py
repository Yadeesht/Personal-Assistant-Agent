"""The terminal input line.

Typing "/" pops up the list of commands with what each one does, on top of the screen
(nothing is printed); it narrows as you type, and Tab or the arrow keys pick one. A
command is highlighted while typed (unknown ones in red), and a hint bar sits at the
bottom while waiting for input. Falls back to plain input() when not in a terminal.
"""

import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

# Every slash command and what it does: the pop-up list and /help both use this.
COMMANDS = {
    "/new": "Save this chat to memory and start a fresh one",
    "/thread": "Show the current chat, or switch: /thread <id>",
    "/reload": "Re-read config/prompts.py without restarting",
    "/clear": "Clear the screen",
    "/help": "List all commands",
    "/exit": "Save this chat to memory and quit",
}


class SlashCommandCompleter(Completer):
    """Offers the commands while the first word starts with "/"."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        typed = text.lower()
        for command, description in COMMANDS.items():
            if command.startswith(typed):
                yield Completion(command, start_position=-len(text), display_meta=description)


class SlashCommandLexer(Lexer):
    """Colours a command as it is typed: known commands in blue, unknown ones in red."""

    def lex_document(self, document):
        def line_fragments(lineno):
            line = document.lines[lineno]
            if lineno == 0 and line.startswith("/"):
                word, separator, rest = line.partition(" ")
                style = "class:command" if word.lower() in COMMANDS else "class:command.unknown"
                return [(style, word), ("", separator + rest)]
            return [("", line)]

        return line_fragments


STYLE = Style.from_dict(
    {
        "prompt": "bold #D97757",
        "command": "bold #5fd7ff",
        "command.unknown": "#ff5f5f",
        "bottom-toolbar": "noreverse #8a8a8a bg:#1c1c1c",
        "completion-menu.completion": "bg:#262626 #d0d0d0",
        "completion-menu.completion.current": "bg:#D97757 #000000 bold",
        "completion-menu.meta.completion": "bg:#1c1c1c #8a8a8a",
        "completion-menu.meta.completion.current": "bg:#D97757 #000000",
    }
)

PROMPT = [("class:prompt", "You ❯ ")]
TOOLBAR = HTML(" <b>/</b> commands   <b>Tab</b> complete   <b>↑ ↓</b> history   <b>Ctrl+C</b> quit")


def interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def make_session(**kwargs) -> PromptSession:
    return PromptSession(
        completer=SlashCommandCompleter(),
        complete_while_typing=True,
        lexer=SlashCommandLexer(),
        style=STYLE,
        bottom_toolbar=TOOLBAR,
        **kwargs,
    )


def read_line(session: PromptSession) -> str:
    """Read one line. While it waits, anything printed (e.g. a background warning)
    appears above the input line instead of breaking it."""
    with patch_stdout(raw=True):
        # Ctrl+C arrives as a key press and raises KeyboardInterrupt; no signal handler
        # is installed, since this runs on the input thread rather than the main one.
        return session.prompt(PROMPT, handle_sigint=False)
