import langchain_core.tools.base

if not hasattr(langchain_core.tools.base, "TOOL_MESSAGE_BLOCK_TYPES"):
    langchain_core.tools.base.TOOL_MESSAGE_BLOCK_TYPES = (
        "text",
        "image_url",
        "image",
        "json",
        "search_result",
        "custom_tool_call_output",
        "document",
        "file",
    )

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

# The prompt texts are read from the module when the graph is built, so /reload can
# re-read config/prompts.py and rebuild without reloading any code.
import config.prompts as prompt_texts
from core.agent import (
    agent_node_factory,
    code_execution_factory,
    memory_node_factory,
    summerizer_node,
    supervisor_node_factory,
    route_to_agent,
    work_completion,
)
from core.state import (
    State,
    internal_agent_route,
    route_after_code_agent,
    route_after_supervisor,
    route_after_supervisor_tools,
    route_after_communication_tools,
    route_after_planning_tools,
    route_after_document_tools,
    route_after_data_tools,
    route_after_presentation_tools,
    route_after_memory,
    route_start,
    route_to_active_agent,
)
from core.confirm import (
    calls_needing_confirmation,
    confirmation_request,
    declined_messages,
    is_approval,
)
from core.llm import build_llm, build_llm_with_tools
from utils.helper import setup_logger

logger = setup_logger(__name__)


def create_agent_tool_node(tools, messages_key: str):
    """
    Creates a node that runs the prebuilt ToolNode with the scoped messages key,
    then updates BOTH the scoped messages key and the global messages key.
    """
    tool_node = ToolNode(tools=tools, handle_tool_errors=True, messages_key=messages_key)

    async def node(state: State):
        # Calls that cannot be undone (e.g. send_email) wait for the user's yes: the
        # graph pauses here and main.py resumes it with the user's reply.
        gated = calls_needing_confirmation(state.get(messages_key) or [])
        if gated:
            decision = interrupt(confirmation_request(gated)) or {}
            reply = str(decision.get("reply", ""))
            if not is_approval(reply):
                declined = declined_messages(state[messages_key], reply)
                return {messages_key: declined, "messages": declined}

        result = await tool_node.ainvoke(state)

        from langgraph.types import Command

        if isinstance(result, Command):
            return result

        if isinstance(result, list):
            # A single AIMessage can carry multiple tool_calls. If any of
            # them (e.g. route_to_agent, work_completion) returns a Command,
            # ToolNode.ainvoke gives back a list mixing Command objects with
            # plain ToolMessages for the other calls. Every one of them must
            # be applied — returning only the first Command silently drops
            # the rest, leaving their tool_call_ids unanswered and breaking
            # the next LLM request.
            commands = [item for item in result if isinstance(item, Command)]
            plain_messages = [item for item in result if not isinstance(item, Command)]

            if plain_messages:
                commands.append(
                    Command(
                        update={
                            messages_key: plain_messages,
                            "messages": plain_messages,
                        }
                    )
                )

            if commands:
                # LangGraph applies every Command in a returned list.
                return commands

            return {messages_key: [], "messages": []}

        # Single dict result: the newly added ToolMessages live under messages_key.
        if isinstance(result, dict):
            tool_messages = result.get(messages_key, [])
        else:
            tool_messages = []
        return {
            messages_key: tool_messages,
            "messages": tool_messages,
        }

    return node


# Content tools are split between the document, data and presentation agents by tool
# name. Keywords match as substrings: "form" also catches the Sheets conditional-
# formatting tools, which is what routes them to the data agent.
DOCUMENT_TOOL_KEYWORDS = ["doc", "drive", "table", "file"]
DATA_TOOL_KEYWORDS = ["sheet", "form", "spreadsheet", "publish"]
PRESENTATION_TOOL_KEYWORDS = ["presentation", "page", "slide"]


def split_content_tools(content_tools):
    """Split content tools into (document_tools, data_tools, presentation_tools) by tool name."""

    def matches(tool, keywords):
        return any(keyword in tool.name.lower() for keyword in keywords)

    document_tools = [t for t in content_tools if matches(t, DOCUMENT_TOOL_KEYWORDS)]
    data_tools = [t for t in content_tools if matches(t, DATA_TOOL_KEYWORDS)]
    presentation_tools = [
        t for t in content_tools if matches(t, PRESENTATION_TOOL_KEYWORDS)
    ]
    return document_tools, data_tools, presentation_tools


def build_graph(tool_sets, checkpointer):
    supervisor_tools = tool_sets.get("supervisor", [])
    communication_tools = tool_sets.get("communication", [])
    planning_tools = tool_sets.get("planning", [])
    content_tools = tool_sets["content"]

    document_tools, data_tools, presentation_tools = split_content_tools(content_tools)

    logger.info(
        f"🔧 Filtered Tools -> Docs: {len(document_tools)} | Data: {len(data_tools)} | Slides: {len(presentation_tools)}"
    )

    supervisor_tools = list(supervisor_tools) + [route_to_agent]
    communication_tools = list(communication_tools) + [work_completion]
    planning_tools = list(planning_tools) + [work_completion]
    document_tools = list(document_tools) + [work_completion]
    data_tools = list(data_tools) + [work_completion]
    presentation_tools = list(presentation_tools) + [work_completion]

    # One handoff per supervisor turn: parallel route_to_agent calls would hand the task
    # to two workers at once, and only one of them can be the active agent.
    supervisor_llm = build_llm_with_tools(supervisor_tools, parallel_tool_calls=False)
    communication_llm = build_llm_with_tools(communication_tools)
    planning_llm = build_llm_with_tools(planning_tools)
    document_llm = build_llm_with_tools(document_tools)
    presentation_llm = build_llm_with_tools(presentation_tools)
    data_llm = build_llm_with_tools(data_tools)

    communication_agent_node = agent_node_factory(
        llm_with_tools=communication_llm,
        system_prompt=prompt_texts.COMMUNICATION_SYSTEM_PROMPT,
        agent_name="communication_agent",
    )

    planning_agent_node = agent_node_factory(
        llm_with_tools=planning_llm,
        system_prompt=prompt_texts.PLANNING_SYSTEM_PROMPT,
        agent_name="planning_agent",
    )

    # The code agent asks the model for plain text (intent, code), so it gets a model
    # without tools: with the supervisor's tools bound it could answer with a tool call.
    code_agent_node = code_execution_factory(
        llm=build_llm(),
        tool_sets=tool_sets,
        agent_name="code_agent",
    )

    document_agent_node = agent_node_factory(
        llm_with_tools=document_llm,
        system_prompt=prompt_texts.DOCUMENT_SYSTEM_PROMPT,
        agent_name="document_agent",
    )

    presentation_agent_node = agent_node_factory(
        llm_with_tools=presentation_llm,
        system_prompt=prompt_texts.PRESENTATION_SYSTEM_PROMPT,
        agent_name="presentation_agent",
    )

    data_agent_node = agent_node_factory(
        llm_with_tools=data_llm,
        system_prompt=prompt_texts.DATA_SYSTEM_PROMPT,
        agent_name="data_agent",
    )

    memory_update_node = memory_node_factory()

    supervisor_node = supervisor_node_factory(
        llm_with_tools=supervisor_llm,
        system_prompt=prompt_texts.SUPERVISOR_SYSTEM_PROMPT,
        agent_name="supervisor",
    )

    builder = StateGraph(State)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("communication_agent", communication_agent_node)
    builder.add_node("planning_agent", planning_agent_node)
    builder.add_node("code_agent", code_agent_node)
    builder.add_node("summerizer_node", summerizer_node)
    builder.add_node("document_agent", document_agent_node)
    builder.add_node("presentation_agent", presentation_agent_node)
    builder.add_node("data_agent", data_agent_node)
    builder.add_node(
        "communication_tools",
        create_agent_tool_node(communication_tools, "communication_messages"),
    )
    builder.add_node(
        "planning_tools",
        create_agent_tool_node(planning_tools, "planning_messages"),
    )
    builder.add_node(
        "supervisor_tools",
        create_agent_tool_node(supervisor_tools, "supervisor_messages"),
    )
    builder.add_node(
        "document_tools",
        create_agent_tool_node(document_tools, "document_messages"),
    )
    builder.add_node(
        "presentation_tools",
        create_agent_tool_node(presentation_tools, "presentation_messages"),
    )
    builder.add_node(
        "data_tools",
        create_agent_tool_node(data_tools, "data_messages"),
    )

    builder.add_node("memory_update_node", memory_update_node)

    builder.add_conditional_edges(
        source=START,
        path=route_start,
        path_map={
            "communication_agent": "communication_agent",
            "planning_agent": "planning_agent",
            "document_agent": "document_agent",
            "presentation_agent": "presentation_agent",
            "data_agent": "data_agent",
            # The user's reply to a code approval request resumes in code_agent.
            "code_agent": "code_agent",
            "summerizer_node": "summerizer_node",
            "memory_update_node": "memory_update_node",
            "supervisor": "supervisor",
        },
    )

    # After the memory and summary steps the message goes to the agent the user is
    # talking to (a worker that asked a question, or the code agent waiting for
    # approval), not always to the supervisor.
    active_agents = {
        "communication_agent": "communication_agent",
        "planning_agent": "planning_agent",
        "document_agent": "document_agent",
        "presentation_agent": "presentation_agent",
        "data_agent": "data_agent",
        "code_agent": "code_agent",
        "supervisor": "supervisor",
    }
    builder.add_conditional_edges(
        "memory_update_node",
        route_after_memory,
        {**active_agents, "summerizer_node": "summerizer_node"},
    )
    builder.add_conditional_edges("summerizer_node", route_to_active_agent, active_agents)

    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "communication_agent": "communication_agent",
            "planning_agent": "planning_agent",
            "document_agent": "document_agent",
            "presentation_agent": "presentation_agent",
            "data_agent": "data_agent",
            "code_agent": "code_agent",
            "supervisor_tools": "supervisor_tools",
            "supervisor": "supervisor",  # for tool fail fallback to same node and ask the LLM to re-decide
            "FINISH": END,
        },
    )

    builder.add_conditional_edges(
        "supervisor_tools",
        route_after_supervisor_tools,
        {
            "communication_agent": "communication_agent",
            "planning_agent": "planning_agent",
            "document_agent": "document_agent",
            "presentation_agent": "presentation_agent",
            "data_agent": "data_agent",
            "code_agent": "code_agent",
            "supervisor": "supervisor",
        },
    )

    builder.add_conditional_edges(
        "code_agent",
        route_after_code_agent,
        {
            "supervisor": "supervisor",
            "END": END,
        },
    )

    builder.add_conditional_edges(
        "communication_agent",
        internal_agent_route,
        {
            "tools": "communication_tools",
            "supervisor": "supervisor",
            "END": END,
        },
    )

    builder.add_conditional_edges(
        "communication_tools",
        route_after_communication_tools,
        {
            "communication_agent": "communication_agent",
            "supervisor": "supervisor",
        },
    )

    builder.add_conditional_edges(
        "planning_agent",
        internal_agent_route,
        {
            "tools": "planning_tools",
            "supervisor": "supervisor",
            "END": END,
        },
    )

    builder.add_conditional_edges(
        "planning_tools",
        route_after_planning_tools,
        {
            "planning_agent": "planning_agent",
            "supervisor": "supervisor",
        },
    )

    builder.add_conditional_edges(
        "document_agent",
        internal_agent_route,
        {"tools": "document_tools", "supervisor": "supervisor", "END": END},
    )

    builder.add_conditional_edges(
        "document_tools",
        route_after_document_tools,
        {
            "document_agent": "document_agent",
            "supervisor": "supervisor",
        },
    )

    builder.add_conditional_edges(
        "data_agent",
        internal_agent_route,
        {"tools": "data_tools", "supervisor": "supervisor", "END": END},
    )

    builder.add_conditional_edges(
        "data_tools",
        route_after_data_tools,
        {
            "data_agent": "data_agent",
            "supervisor": "supervisor",
        },
    )

    builder.add_conditional_edges(
        "presentation_agent",
        internal_agent_route,
        {"tools": "presentation_tools", "supervisor": "supervisor", "END": END},
    )

    builder.add_conditional_edges(
        "presentation_tools",
        route_after_presentation_tools,
        {
            "presentation_agent": "presentation_agent",
            "supervisor": "supervisor",
        },
    )

    graph = builder.compile(checkpointer=checkpointer)
    return graph
