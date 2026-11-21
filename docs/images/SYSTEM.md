# JARVIS: A Glimpse into the Future of AI Assistants

JARVIS is a sophisticated, multi-agent AI system designed to be a proactive and intelligent assistant. It is built on a modular, distributed architecture that allows for easy expansion and integration of new capabilities. This document provides a comprehensive overview of the JARVIS system, its architecture, features, and how to use it.

## System Architecture

JARVIS's architecture is designed for robustness and scalability. It is composed of several key components that work together to provide a seamless user experience.

### Modular Tool Registry (In-Process Services)

At its core, JARVIS uses a modular tool registry structure to organize different services. This allows for a clean separation of concerns where different agents access distinct sets of tools, each representing a specific domain. By loading all tool sets in-process directly within the runtime, JARVIS achieves maximum execution speed and avoids the latency or reliability issues of running separate external server processes. The current implementation includes the following tool categories:

*   **Communication Tools**: Handles Gmail and Google Chat API interactions.
*   **Content Tools**: Manages Google Drive, Docs, Sheets, Slides, and Forms.
*   **Planning Tools**: Handles Google Calendar and Google Tasks.
*   **Supervisor Tools**: Manages search, RAG, and core orchestrations.

### Graph-Based Conversation Flow (LangGraph)

JARVIS employs a stateful, graph-based approach to manage complex conversational flows, built using the `langgraph` library. This allows for dynamic decision-making and routing between specialized agents. The main nodes in the graph are:

*   **Supervisor**: The central routing node. It analyzes the user's intent and the conversation history to decide which specialized agent or tool should handle the next step.
*   **Specialized Agents**: These are the `communication_agent`, `planning_agent`, `content_agent`, and `code_agent`. Each is a dedicated entity with specific tools and expertise.
*   **Tools (ToolNode)**: These nodes execute the specific tools (e.g., Gmail API, Google Calendar API) available to the respective agents.
*   **Summerizer**: A dedicated node that condenses long conversation histories to maintain context within LLM token limits, ensuring efficient and relevant interactions.
*   **Memory Update Node**: This crucial node manages the system's long-term memory, integrating new information into both the Episodic RAG and the Knowledge Graph.

The routing logic within the graph is dynamic:
*   **`route_start`**: Determines the initial step for a new user input. It checks for needed memory updates (e.g., if a new day has started or if the conversation is too long), potential clarification re-routes, or token limits for summarization, before defaulting to the Supervisor.
*   **`internal_agent_route`**: Governs the flow within a sub-agent. It routes to tools if the agent needs them, or back to the Supervisor if the agent has a `FINAL ANSWER`, `CLARIFICATION NEEDED`, or `TALK TO USER` message.
*   **`route_after_supervisor`**: Directs the flow from the Supervisor to the chosen specialized agent, tools, or concludes the conversation (`FINISH`).

### Specialized Agents

JARVIS's functionality is driven by a set of specialized agents, each powered by an LLM and equipped with specific tools and system prompts that define their behavior and capabilities.

*   **Supervisor Agent**: The master orchestrator. It directs traffic to the appropriate sub-agent based on user intent. It also includes built-in knowledge graph and web search tools, which it uses only when explicitly instructed by the user.
*   **Communication Agent**: Dedicated to handling all communication tasks, such as sending emails (Gmail), managing Google Chat conversations, and reading messages. It adheres to a strict output format, providing detailed "Task Receipts" for every completed action.
*   **Planning Agent**: Manages scheduling, reminders, and tasks. It interacts with Google Calendar and Google Tasks APIs to create, modify, or delete events and tasks. Like other agents, it follows a strict output format for clarity.
*   **Content Agent**: Responsible for managing and generating content across Google Drive, Docs, Sheets, Slides, and Forms. It can search, create, update, and share files, ensuring adherence to specific output formats and detailed "Task Receipts."
*   **Code Agent**: A highly capable agent designed for complex, programmatic tasks. It's invoked for automation, batch processing, data manipulation, or when tasks require processing large amounts of information efficiently. It uses a secure sandboxed environment for code execution.

### Memory Management

JARVIS implements advanced memory management systems for continuous learning and context awareness.

*   **Episodic Retrieval-Augmented Generation (RAG)**: This system processes conversational logs from the `MEMORY_DB`, cleans them, intelligently chunks them based on token limits and time gaps, embeds them using the `gte-modernbert-base` model, and stores them in a Qdrant vector database (`EPISODIC_RAG_DB`). For retrieval, it can reconstruct multi-part task contexts and expand related chunks, providing rich, temporally organized conversational history for context.
*   **Knowledge Graph**: Built on KuzuDB (`KNOWLEDGE_GRAPH_DB`), this component stores structured information (entities and relationships) extracted from conversations. It uses the `bge-small` embedding model for semantic search. LLMs are leveraged for:
    *   **Extraction**: Identifying and formalizing new entities and relationships from chat logs.
    *   **Validation & Reconciliation**: Merging new knowledge with existing graph data, preventing duplicates, and updating information to ensure a consistent and evolving knowledge base.
    *   It supports CRUD operations for entities and relationships and can visualize the graph using NetworkX and Matplotlib.

## Features

JARVIS is packed with a wide range of features that make it a powerful and versatile AI assistant.

*   **Dynamic Code Execution (Sandboxed)**: The Code Agent can dynamically generate Python code based on user requests and execute it safely within an isolated sacrificial sandbox process. This allows JARVIS to perform complex data manipulations, integrate with various APIs programmatically, and automate multi-step workflows.
*   **Persistent & Contextual Memory**: Through its Episodic RAG and Knowledge Graph, JARVIS learns from every interaction, building a rich, searchable memory of past conversations, entities, and relationships. This enables highly personalized and context-aware responses.
*   **Advanced Tool Integration**: Each agent is equipped with a specific set of tools (e.g., Google Workspace APIs). The system provides structured methods for the LLMs to understand and utilize these tools effectively.
*   **Configurable LLM Backends**: JARVIS is designed to work with various LLM providers (OpenRouter, Groq, Hugging Face) and models, configurable via `config/settings.py`, allowing flexibility and performance tuning.
*   **Conversation Summarization**: Long conversations are automatically summarized to maintain context and improve efficiency for LLM processing.

## How to Use

Interacting with JARVIS is designed to be simple and intuitive. You can communicate with it using text input.

When making a request, provide clear and specific instructions. The more precise your request, the better JARVIS's Supervisor Agent can route it to the appropriate specialized agent or tool to achieve the desired outcome.

Here are a few examples of how you can interact with JARVIS:

*   "Send an email to John Doe at `john.doe@example.com` with the subject 'Project Update' and the body 'Hi John, the files for the Q3 project are ready for review. Best regards, Yadeesh.'" (Utilizes Communication Agent)
*   "Schedule a 1-hour meeting with the marketing team for tomorrow at 2 PM about the new campaign launch, and add Alice and Bob as attendees." (Utilizes Planning Agent)
*   "Create a new Google Doc titled 'Meeting Notes - January 22nd' in my Drive and add a bulleted list: 'Discuss Q1 results', 'Plan Q2 strategy', 'Assign action items'." (Utilizes Content Agent)
*   "Write a Python script that reads all CSV files in my Google Drive folder named 'Sales Data', calculates the total revenue for each month, and saves the results into a new Google Sheet named 'Monthly Sales Summary'." (Utilizes Code Agent)

## How This is Different

JARVIS stands apart from conventional AI assistants due to its sophisticated architecture and focus on deep contextual understanding and dynamic execution.

*   **True Multi-Agent Collaboration**: Unlike single-model chatbots, JARVIS leverages a dynamic network of specialized agents that collaborate, each excelling in its domain (communication, planning, content, code). This enables handling complex, multi-faceted requests that would overwhelm a monolithic AI.
*   **Proactive & Context-Aware Intelligence**: Beyond merely reacting to queries, JARVIS actively learns and retains context through its Episodic RAG and Knowledge Graph. It doesn't just process current input but understands the ongoing narrative, remembers past interactions, and uses this deep memory to provide more relevant, personalized, and even proactive assistance.
*   **Dynamic Code Generation & Secure Execution**: The ability to write and execute custom Python code within a sandboxed environment transforms JARVIS from a mere tool-caller into a powerful programmable assistant. This capability allows it to tackle bespoke automation tasks, complex data processing, and novel problem-solving on the fly, pushing the boundaries of what an AI assistant can achieve.
*   **Robust & Scalable Architecture**: The modular tool registry and LangGraph-based conversation flow provide a highly resilient and scalable foundation. This architecture allows for seamless integration of new agents, tools, and LLM providers, ensuring JARVIS can continuously evolve and adapt to new demands without requiring a complete overhaul.

JARVIS represents a significant leap forward in AI assistance, offering a powerful, flexible, and intelligent system that can revolutionize the way we interact with technology.
