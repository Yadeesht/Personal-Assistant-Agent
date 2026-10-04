import json
import re
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Callable
from utils.helper import setup_logger
from langchain_core.messages import BaseMessage
from core.state import TaskSpec


logger = setup_logger(__name__)


class CodeExecutionAgent:
    def __init__(self, llm_client, tool_sets: Dict[str, Any]):
        """Initialize code agent with LLM client, tool registry, and sandbox path."""
        self.llm = llm_client
        self.tool_sets = tool_sets
        self.sandbox_path = Path(__file__).parent / "sandbox"
        self.sandbox_path.mkdir(exist_ok=True)

    async def execute_workflow(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        task_spec = await self._resolve_intent(messages)

        logger.info(f"Task Specs: {task_spec}")

        query_acceptence = True if task_spec.required_tools_hint else False

        if not query_acceptence:
            logger.warning(
                "Query was rejected by intent resolver. Due to unclear intent or no tools required."
            )
            return {
                "status": "error",
                "summary": "Query rejected due to unclear intent or no tools required",
            }

        tool_map = self._create_tool_map(task_spec.required_tools_hint)

        tool_schemas = await self._load_tool_schemas(task_spec.required_tools_hint)

        code_prompt = self._build_code_generation_prompt(task_spec, tool_schemas)

        generated_code = await self._generate_code(code_prompt)

        result = await self._execute_in_sandbox(generated_code, tool_map)

        result["generated_code"] = generated_code
        result["task_goal"] = task_spec.primary_goal

        return result

    def _create_tool_map(self, required_hints: List[str]) -> Dict[str, Callable]:
        tool_map = {}
        active_tools = []
        for category, tools in self.tool_sets.items():
            if not required_hints or category in required_hints:
                active_tools.extend(tools)

        for tool in active_tools:

            def make_wrapper(t):
                async def wrapper(**kwargs):
                    logger.info(f"⚙️ Executing Tool: {t.name}")
                    logger.info(f"   Parameters: {kwargs}")
                    try:
                        if hasattr(t, "ainvoke"):
                            result = await t.ainvoke(kwargs)
                        elif hasattr(t, "arun"):
                            result = await t.arun(kwargs)
                        elif hasattr(t, "run"):
                            result = t.run(kwargs)
                        else:
                            result = t(kwargs)

                        logger.info(f"   Raw result type: {type(result)}")
                        logger.info(f"   Raw result: {str(result)}")

                        # Parse string results into dict
                        if isinstance(result, str):
                            import json

                            try:
                                result = json.loads(result)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Could not parse as JSON, wrapping in dict"
                                )
                                result = {
                                    "success": True,
                                    "data": result,
                                    "raw_output": result,
                                }

                        return result

                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Tool {t.name} failed: {error_msg}")

                        return {
                            "error": error_msg,
                            "success": False,
                            "count": 0,
                            "emails": [],
                            "message": f"Tool execution failed: {error_msg[:200]}",
                        }

                return wrapper

            tool_map[tool.name] = make_wrapper(tool)

        logger.info(f"🔧 Created tool map with {len(tool_map)} tools")
        return tool_map



    async def _resolve_intent(self, messages: List[BaseMessage]) -> TaskSpec:
        system_prompt = """
            You are a task analyzer. Your job is to extract structured intent from the conversation
            and determine which tools are STRICTLY REQUIRED to complete the user's LATEST request.

            Return a JSON object with EXACTLY this structure:
            {
            "primary_goal": "Clear, concise description of what the user wants to accomplish",
            "required_tools_hint": ["communication", "planning", "content"],
            "context_variables": {"key": "string value only"},
            "last_error": null
            }

            ### TOOL SELECTION RULES (CRITICAL)

            You MUST include a tool in "required_tools_hint" ONLY if the user's request
            CANNOT be completed without that tool.

            #### Tool eligibility rules:

            1. "communication"
            Include ONLY IF the user explicitly asks to:
            - communicate with a person via email or chat

            2. "planning"
            Include ONLY IF the user explicitly asks to:
            - create, update, delete, or view calendar events
            - schedule or reschedule meetings
            - manage tasks or reminders (Google Tasks)

            3. "content"
            Include ONLY IF the user explicitly asks to:
            - create, edit, read, search, or share files
            - work with Google Docs, Sheets, Slides, Forms
            - access Google Drive content

            4. If NONE of the above conditions are met:
            - "required_tools_hint" MUST be an empty list []

            ### IMPORTANT CONSTRAINTS

            - Base your decision BASED ON user request
            - If the request can be answered by reasoning alone, return an empty tool list
            - Never infer tools from intent like "prepare", "think", "draft", or "explain"
            - If the user asks for advice or explanation ONLY, no tools are required

            ### CONTEXT VARIABLES

            - Include only values that are REQUIRED for downstream execution
            - ALL values MUST be strings (no numbers, booleans, objects, or arrays)
            """

        recent_messages = messages
        conversation = "\n".join([f"{m.type}: {m.content}" for m in recent_messages])

        prompt = f"{system_prompt}\n\nConversation:\n{conversation}"

        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])

        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))

                sanitized = {
                    "primary_goal": data.get("primary_goal", "Unknown goal"),
                    "required_tools_hint": data.get(
                        "required_tools_hint", ["communication"]
                    ),
                    "context_variables": {},
                    "last_error": data.get("last_error")
                    if data.get("last_error")
                    else None,
                }

                ctx = data.get("context_variables", {})
                if isinstance(ctx, dict):
                    sanitized["context_variables"] = {k: str(v) for k, v in ctx.items()}

                return TaskSpec(**sanitized)
            except Exception as e:
                logger.error(f"TaskSpec creation failed: {e}")

        return TaskSpec(
            primary_goal=recent_messages[-1].content if recent_messages else "No goal",
            required_tools_hint=["communication"],
            context_variables={},
            last_error=None,
        )

    async def _load_tool_schemas(self, tool_types: List[str]) -> Dict[str, Any]:
        schemas = {}
        for tool_type in tool_types:
            if tool_type in self.tool_sets:
                tools = self.tool_sets[tool_type]

                tool_schemas = []
                for t in tools:
                    # Debug: See what attributes the tool has
                    logger.debug(f"Tool: {t.name}, Type: {type(t).__name__}")
                    logger.debug(
                        f"  Attributes: {[a for a in dir(t) if not a.startswith('_')]}"
                    )

                    schema = {
                        "name": t.name,
                        "description": t.description,
                    }

                    # Handle different tool types
                    if hasattr(t, "inputSchema"):
                        schema["parameters"] = t.inputSchema
                    elif hasattr(t, "args_schema") and t.args_schema:
                        # ✅ Check if it's already a dict or a Pydantic model
                        if isinstance(t.args_schema, dict):
                            schema["parameters"] = t.args_schema
                        else:
                            # It's a Pydantic model, call .schema()
                            schema["parameters"] = t.args_schema.schema()
                    elif hasattr(t, "args"):
                        schema["parameters"] = t.args
                    else:
                        schema["parameters"] = {}
                        logger.warning(f"Tool {t.name} has no schema attribute")

                    tool_schemas.append(schema)  # ✅ Add this line - it was missing!

                schemas[tool_type] = tool_schemas  # ✅ This was indented wrong

        logger.info(f"📦 Loaded schemas for {len(schemas)} tool types")
        for tool_type, tools in schemas.items():
            logger.info(f"   {tool_type}: {len(tools)} tools")

        return schemas

    def _build_code_generation_prompt(self, spec: TaskSpec, schemas: Dict) -> str:
        """
        Builds the prompt using the Clean Spec.
        """
        return f"""
        You are an expert Python developer.
        
        GOAL: {spec.primary_goal}
        
        CONTEXT VARIABLES:
        {spec.context_variables}
        
        PREVIOUS ERROR (Fix this if present):
        {spec.last_error or "None"}
        
        AVAILABLE TOOLS (ALL ARE ASYNC):
        {self._format_schemas(schemas)}

        Requirements:
        1. You CANNOT use `input()` or interactive commands.
        2. You MUST use the provided tools for external actions.
        3. ALL TOOL CALLS MUST BE AWAITED: `result = await tool_name(**params)`
        4. ALWAYS CHECK FOR ERRORS in tool results before processing.
        5. Return a dictionary with 'summary', 'details', 'artifacts'.
        
        Example error handling:
        ```python
        result = await get_unread_emails(date=0)
        
        # ✅ Always check for errors
        if result.get("error") or not result.get("success", True):
            return {{
                "summary": f"Failed: {{result.get('error', 'Unknown error')}}",
                "details": {{"error": result.get("error")}},
                "artifacts": []
            }}
        
        # Now safe to use result
        emails = result.get("emails", [])
        ```
        
        Follow this template strictly:
        ```python
        import asyncio
        import json
        from typing import Dict, Any, List

        async def execute_workflow() -> Dict[str, Any]:
            # Step 1: Call tool with await
            result = await tool_name(param1="value")

            # Step 2: Check for errors
            if result.get("error"):
                return {{
                    "summary": f"Tool failed: {{result['error']}}",
                    "details": {{"error": result["error"]}},
                    "artifacts": []
                }}

            # Step 3: Process successful results
            data = result.get("data", [])

            return {{
                "summary": f"Successfully processed {{len(data)}} items",
                "details": {{"count": len(data), "items": data}},
                "artifacts": []
            }}

        if __name__ == "__main__":
            result = asyncio.run(execute_workflow())
            # Must print valid JSON (not a Python dict repr) so the caller can parse it.
            print(json.dumps(result, default=str))
        ```
        """

    async def _generate_code(self, prompt: str) -> str:
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        code = self._extract_code_block(response.content)

        logger.info(f"Generated code:\n{code}")
        return code

    async def _ensure_sandbox_server_running(self):
        import socket
        def is_port_open(port: int) -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('127.0.0.1', port)) == 0
                
        if not is_port_open(9000):
            logger.info("Sandbox server not running on port 9000. Spawning it...")
            sandbox_script = Path(__file__).resolve().parent / "sandbox_server.py"
            import sys
            import subprocess
            subprocess.Popen(
                [sys.executable, str(sandbox_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True
            )
            await asyncio.sleep(1.0)

    def _prepend_tool_wrappers(self, code: str, tool_map: Dict[str, Callable]) -> str:
        if not tool_map:
            return code
            
        wrappers = [
            "import httpx",
            "import asyncio",
            "from typing import Dict, Any, List",
            ""
        ]
        
        for tool_name in tool_map.keys():
            wrapper_func = f"""
async def {tool_name}(**kwargs):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "http://127.0.0.1:8080/api/sandbox/execute_tool",
                json={{"tool_name": "{tool_name}", "arguments": kwargs}},
            )
            if r.status_code == 200:
                return r.json()
            else:
                return {{"error": f"Tool HTTP error: {{r.status_code}}", "success": False}}
    except Exception as e:
        return {{"error": f"Tool connection failed: {{str(e)}}", "success": False}}
"""
            wrappers.append(wrapper_func)
            
        return "\n".join(wrappers) + "\n\n" + code

    async def _execute_in_sandbox(self, code: str, tool_map: Dict[str, Callable] = None) -> Dict[str, Any]:
        """Execute in local Python Code Sandbox Server on Port 9000"""
        import httpx
        from core.tool_bridge import ensure_bridge_running, set_active_tool_map

        await self._ensure_sandbox_server_running()

        # The sandbox subprocess calls back into this process (via the tool
        # bridge on port 8080) to actually invoke tools that need live
        # credentials/services held here.
        set_active_tool_map(tool_map)
        await ensure_bridge_running()

        full_code = self._prepend_tool_wrappers(code, tool_map)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:9000/execute",
                    json={"code": full_code},
                    timeout=35.0
                )
                
            if response.status_code == 200:
                res = response.json()
                if res.get("status") == "success":
                    stdout = res.get("stdout", "").strip()
                    try:
                        json_match = re.search(r"\{.*\}", stdout, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group(0))
                        else:
                            result = json.loads(stdout)
                            
                        return {
                            "status": "success",
                            "summary": result.get("summary", "Completed"),
                            "full_output": result,
                        }
                    except Exception as parse_error:
                        logger.error(f"Failed to parse stdout as JSON: {stdout}. Error: {parse_error}")
                        return {
                            "status": "success",
                            "summary": f"Completed (unstructured output): {stdout[:200]}",
                            "full_output": {"stdout": stdout},
                        }
                else:
                    error_msg = res.get("stderr", "") or res.get("stdout", "")
                    logger.error(f"Execution failed: {error_msg}")
                    return {
                        "status": "error",
                        "error": error_msg,
                        "summary": f"Failed: {error_msg[:200]}",
                    }
            else:
                return {
                    "status": "error",
                    "error": f"Sandbox server returned status code {response.status_code}",
                    "summary": "Sandbox server error",
                }
                
        except Exception as e:
            logger.error(f"Sandbox communication error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "summary": f"Execution error: {e}",
            }

    def _format_schemas(
        self, schemas: Dict
    ) -> str:  # need to check if they really better than the basic json transfer
        formatted = []
        for tool_type, tools in schemas.items():
            formatted.append(f"\n# --- {tool_type.upper()} TOOLS ---")

            for tool in tools:
                name = tool["name"]
                desc = tool["description"]
                params = tool.get("parameters", {})
                props = params.get("properties", {})
                required = params.get("required", [])

                args_list = []
                for prop_name, prop_info in props.items():
                    prop_type = prop_info.get("type", "any")
                    if prop_name not in required:
                        args_list.append(
                            f"{prop_name}: {prop_type} = None"
                        )  # handle llm to know is it required or not
                    else:
                        args_list.append(f"{prop_name}: {prop_type}")

                args_str = ", ".join(args_list)
                signature = f'async def {name}({args_str}):\n    """{desc}"""'

                formatted.append(signature)

        return "\n".join(formatted)

    def _extract_code_block(self, response: str) -> str:
        """Extract code from markdown code blocks"""
        import re

        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        return response
