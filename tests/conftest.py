"""Shared test setup: loaded by pytest before any test module."""

import os
import tempfile
from pathlib import Path

# Background-thread logs go to data/logs/memory.log in a real run; keep the tests'
# memory saves out of the user's log file.
os.environ["MEMORY_LOG_FILE"] = str(Path(tempfile.gettempdir()) / "personal-assistant-test-memory.log")
