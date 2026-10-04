from langchain_core.tools import tool
from typing import List

class ToolRegistry:
    def __init__(self, name: str):
        self.name = name
        self._tools = []

    def tool(self, *args, **kwargs):
        def decorator(func):
            # Wrap the function in LangChain's native tool decorator
            lc_tool = tool(*args, **kwargs)(func)
            self._tools.append(lc_tool)
            return func
        return decorator

    def list_tools(self) -> List:
        return self._tools

# Communication Server
communication_server = ToolRegistry("Communication Server")

# Planning Server
planning_server = ToolRegistry("Planning Server")

# Content Server
content_server = ToolRegistry("Content Server")

# Supervisor Server
supervisor_server = ToolRegistry("Supervisor Server")

__all__ = [
    "communication_server",
    "planning_server",
    "content_server",
    "supervisor_server",
]
