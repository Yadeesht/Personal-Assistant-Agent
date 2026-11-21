"""
In-process HTTP bridge for the sandboxed code agent.

Generated code runs as a standalone subprocess spawned by
`core/sandbox_server.py` (port 9000). That subprocess has no access to the
live LangChain tool objects in the main process (they hold open Google API
credentials, event loops, etc.), so it can't call them directly.

Instead, the generated code calls thin async wrapper functions
(`core.codeagent.CodeExecutionAgent._prepend_tool_wrappers`) that POST to
this bridge. The bridge looks the requested tool up in the tool map that is
currently active for the in-flight sandbox run and invokes it here, in the
main process, then returns the JSON result to the subprocess.
"""

import asyncio
from typing import Any, Callable, Dict, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from utils.helper import setup_logger

logger = setup_logger(__name__)

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = 8080

app = FastAPI(title="JARVIS Sandbox Tool Bridge")

# The tool map for the sandbox run currently in flight. The code agent only
# ever executes one workflow at a time in this codebase, so a module-level
# reference (swapped in right before each run) is sufficient.
_active_tool_map: Dict[str, Callable] = {}

_server: Optional[uvicorn.Server] = None
_server_task: Optional[asyncio.Task] = None
_start_lock = asyncio.Lock()


class ToolExecuteRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = {}


@app.post("/api/sandbox/execute_tool")
async def execute_tool(req: ToolExecuteRequest):
    wrapper = _active_tool_map.get(req.tool_name)
    if wrapper is None:
        return {"error": f"Unknown tool '{req.tool_name}'", "success": False}
    try:
        return await wrapper(**req.arguments)
    except Exception as e:
        logger.error(f"Tool bridge execution failed for '{req.tool_name}': {e}")
        return {"error": str(e), "success": False}


def set_active_tool_map(tool_map: Optional[Dict[str, Callable]]) -> None:
    """Point the bridge at the tool map for the sandbox run about to start."""
    global _active_tool_map
    _active_tool_map = tool_map or {}


async def ensure_bridge_running() -> None:
    """Start the tool bridge server on this event loop if it isn't already running."""
    global _server, _server_task

    if _server_task is not None and not _server_task.done():
        return

    async with _start_lock:
        if _server_task is not None and not _server_task.done():
            return

        config = uvicorn.Config(
            app,
            host=BRIDGE_HOST,
            port=BRIDGE_PORT,
            log_level="warning",
            loop="asyncio",
        )
        _server = uvicorn.Server(config)
        _server_task = asyncio.create_task(_server.serve())

        # Wait for the socket to actually be bound before returning, so the
        # first tool call from the sandbox subprocess doesn't race startup.
        for _ in range(50):
            if _server.started:
                break
            await asyncio.sleep(0.1)

        if not _server.started:
            logger.warning("Tool bridge server did not report ready in time.")
        else:
            logger.info(
                f"🌉 Sandbox tool bridge listening on {BRIDGE_HOST}:{BRIDGE_PORT}"
            )
