from rag.knowledge_graph import KnowledgeGraph
from rag.episodic_rag import EpisodicRAG
import sys
import functools
import threading
from datetime import datetime
from pathlib import Path
import asyncio

from app_tools.core.server_init import supervisor_server
from config.settings import MEMORY_RECALL
from utils.helper import setup_logger
from utils.memory_manager import add_instruction, list_instructions, remove_instruction
import logging

logger = setup_logger(__name__)

_kg_instance = None
_rag_instance = None
_instance_lock = threading.Lock()


class _ThreadSafeKG:
    """The shared KnowledgeGraph, safe to use from several threads.

    Memory is saved on a background thread while recall and the knowledge-graph tools
    run on others, and they share one kuzu connection and one embedding model. Every
    graph call holds a lock; the two LLM-only calls (extraction and validation) run
    without it, so a slow model reply during saving does not hold up recall.
    """

    _UNLOCKED = {"generate_entity_relation", "validate_entity_relation"}

    def __init__(self, kg: KnowledgeGraph):
        self._kg = kg
        self._lock = threading.RLock()

    def __getattr__(self, name):
        attr = getattr(self._kg, name)
        if not callable(attr) or name in self._UNLOCKED:
            return attr

        @functools.wraps(attr)
        def locked(*args, **kwargs):
            with self._lock:
                return attr(*args, **kwargs)

        return locked


def get_rag_instance():
    global _rag_instance
    with _instance_lock:
        if _rag_instance is None:
            try:
                _rag_instance = EpisodicRAG()
            except Exception as e:
                logger.error(f"Failed to initialize EpisodicRAG: {e}")
                raise
    return _rag_instance


def get_kg_instance():
    global _kg_instance
    with _instance_lock:
        if _kg_instance is None:
            try:
                _kg_instance = _ThreadSafeKG(KnowledgeGraph())
            except Exception as e:
                logger.error(f"Failed to initialize KnowledgeGraph: {e}")
                raise
    return _kg_instance


# --- Recall: knowledge-graph facts looked up for each user message ------------------
# Past conversations (episodic memory) are not recalled automatically: a new chat should
# not drift towards old ones. The supervisor looks them up with retrieve_relevant_chunks
# only when the user asks about an earlier conversation.
# Scores are cosine similarities from the graph's embedding model (bge-small). They were
# set without real data to compare against; adjust them once memory has been in use.
RECALL_MAX_FACTS = 8
RECALL_MIN_FACT_SCORE = 0.55  # an entity found by meaning (not named in the message)
RECALL_MIN_NEIGHBOUR_SCORE = 0.6  # an entity reached through one found by meaning
RECALL_MIN_WORDS = 3  # shorter messages ("yes", "approve") only match entity names

# --- Profile: the user's own facts, shown with every message ------------------------
PROFILE_ENTITY = "Yadeesh"  # the name the extraction prompt gives the user
PROFILE_MAX_CONNECTIONS = 12

_profile_cache = None  # (text, ids of the entities it shows)
_profile_version = 0  # bumped whenever the graph changes


def _stored(updated_at) -> str:
    return f" (stored {str(updated_at)[:10]})" if updated_at else ""


def _fact_line(row) -> str:
    line = f"- {row.get('id')} ({row.get('type')}): {row.get('description') or ''}"
    return line.rstrip() + _stored(row.get("updated_at"))


def invalidate_profile():
    """Forget the cached profile, so the next message rebuilds it. Call after the
    knowledge graph changes."""
    global _profile_cache, _profile_version
    _profile_version += 1
    _profile_cache = None


def get_profile() -> tuple:
    """Key facts about the user from the knowledge graph: their own entity and the
    people, projects and organizations directly connected to them, newest first.

    Returns (text, ids of the entities shown). Cached until the graph changes, so it
    needs no lookup per message (and no embedding model). ("", set()) when memory is
    off or nothing about the user is stored.
    """
    global _profile_cache
    if not MEMORY_RECALL:
        return "", set()
    if _profile_cache is not None:
        return _profile_cache

    version = _profile_version
    text, ids = "", set()
    try:
        kg = get_kg_instance()
        if kg.count_entities():
            profile = kg.profile(PROFILE_ENTITY, PROFILE_MAX_CONNECTIONS)
            user = profile["entity"]
            if user:
                lines = [_fact_line(user)]
                ids.add(user["id"])
                for c in profile["connections"]:
                    relation = (
                        f"{user['id']} {c['relation_type']} {c['id']}"
                        if c["outgoing"]
                        else f"{c['id']} {c['relation_type']} {user['id']}"
                    )
                    line = f"- {relation}: {c['id']} ({c['type']}) {c['description'] or ''}"
                    lines.append(line.rstrip() + _stored(c.get("updated_at")))
                    ids.add(c["id"])
                text = "\n".join(lines)
    except Exception as e:
        logger.error(f"Building the memory profile failed: {e}")
        return "", set()  # not cached: try again next message

    result = (text, ids)
    if version == _profile_version:  # the graph did not change while building it
        _profile_cache = result
    return result


def _recall_facts(kg, query: str, exclude: set) -> str:
    """Graph facts for a message: entities it names (always), entities close to it in
    meaning (above the score thresholds), and how they are connected."""
    found = {}
    named = [n for n in kg.entities_named_in(query) if n not in exclude]
    for row in kg.entity_facts(named):
        found[row["id"]] = row

    if len(query.split()) >= RECALL_MIN_WORDS:
        hits = kg.find_similar_with_expansion(query)
        if hits is not None and not getattr(hits, "empty", True):
            for row in hits.to_dict("records"):
                if row["id"] in exclude or row["id"] in found:
                    continue
                threshold = (
                    RECALL_MIN_FACT_SCORE
                    if row.get("hops") == 0
                    else RECALL_MIN_NEIGHBOUR_SCORE
                )
                if (row.get("base_score") or 0) >= threshold:
                    found[row["id"]] = row

    if not found:
        return ""
    shown = list(found.values())[:RECALL_MAX_FACTS]
    lines = [_fact_line(row) for row in shown]
    connections = kg.relationships_of([row["id"] for row in shown], RECALL_MAX_FACTS)
    if connections:
        lines.append(
            "Connections: "
            + "; ".join(
                f"{c['source']} {c['relation_type']} {c['target']}" for c in connections
            )
        )
    return "\n".join(lines)


def recall_memory(query: str, exclude_entities=()) -> str:
    """Knowledge-graph facts related to a user message, as a short text block.

    Entities the message names or is about, and how they are connected. Facts about
    exclude_entities (already shown in the profile) are left out. No LLM call. Returns
    "" when recall is off (MEMORY_RECALL) or nothing matches; an empty graph is skipped
    without loading the embedding model. Past conversations are not included (see
    retrieve_relevant_chunks).
    """
    if not MEMORY_RECALL or not query or not query.strip():
        return ""

    try:
        kg = get_kg_instance()
        if kg.count_entities():
            facts = _recall_facts(kg, query, set(exclude_entities))
            if facts:
                return f"Known facts:\n{facts}"
    except Exception as e:
        logger.error(f"Knowledge graph recall failed: {e}")
    return ""


def preload_memory_models():
    """Load the knowledge graph's embedding model (and build the profile) at startup,
    on a background thread, so the first message's recall is not slowed by it. The
    episodic model is not preloaded: past conversations are only looked up on request
    (and memory saves load it in the background)."""
    if not MEMORY_RECALL:
        return
    try:
        kg = get_kg_instance()
        if kg.count_entities():
            kg.warm_up()
            get_profile()
    except Exception as e:
        logger.error(f"Preloading the knowledge graph failed: {e}")


@supervisor_server.tool()
async def retrieve_from_knowledge_graph(query: str):
    """
    USE THIS TOOL TO QUERY THE INTERNAL KNOWLEDGE GRAPH ONLY NOT ANY OTHER TOOL FOR KNOWLEDGE GRAPH
        Query the internal Knowledge Graph to retrieve context about specific entities (Person, Project, Organization, Tool, Concept, Event, Resource).

        Args:
            query: Search string or entity reference used to locate related knowledge graph nodes.
    """
    try:
        kg = get_kg_instance()
        results = await asyncio.to_thread(kg.find_similar_with_expansion, query)
        await asyncio.to_thread(kg.visualize)
        return results
    except Exception as e:
        logger.error(f"Error retrieving from knowledge graph: {e}")
        return f"Error retrieving from knowledge graph: {e}"


@supervisor_server.tool()
async def add_information_to_knowledge_graph(details: str):
    """
    Persist new information as structured knowledge within the internal Knowledge Graph for long-term contextual retrieval.

    Args:
        details: Text description containing facts, relationships, or updates to be stored.
    """
    try:
        # The same extraction and validation as memory saves (core.agent.store_facts),
        # with a note that the user asked for this to be stored.
        from core.agent import store_facts

        kg = get_kg_instance()
        learned_from = f"told directly by Yadeesh on {datetime.now():%Y-%m-%d}"
        result = await asyncio.to_thread(
            store_facts, kg, f"Yadeesh (user): {details}", learned_from, True
        )
        if result["status"] == "skipped":
            return "Could not store it: the knowledge-graph extraction failed. Please try again."
        if result["status"] == "nothing_new":
            return "Nothing new to store: these facts are already in the knowledge graph."
        invalidate_profile()
        await asyncio.to_thread(kg.visualize)
        return (
            f"Stored in the knowledge graph: {result['entities']} entities and "
            f"{result['relationships']} relationships created or updated."
        )
    except Exception as e:
        logger.error(f"Error adding information to knowledge graph: {e}")
        return f"Error adding information to knowledge graph: {e}"


@supervisor_server.tool()
async def retrieve_relevant_chunks(query: str, top_k: int = 5, conditions: dict = None):
    """
    USE THIS TOOL TO SEARCH CONVERSATION HISTORY WITH EPISODIC RAG ONLY THIS IS NOT KNOWLEDGE GRAPH TOOL

    Search conversation history using Episodic RAG to retrieve past interactions, decisions, and contextual information.

    Past conversations are NOT shown to you automatically. Call this when SIR refers to an
    earlier conversation: asks when or whether something was discussed ("I don't remember
    when we talked about...", "what did we decide about..."), asks you to look something up
    from earlier chats, or refers to it ("like last time", "the plan we discussed").
    Each result includes `timestamp` (when it was discussed) and `thread_id` (which chat).

    Args:
        query: Descriptive search query (e.g., "bug fix discussion", "API project status", "email content about deadline")
        top_k: Number of relevant conversation chunks to retrieve (default: 5)
        conditions: Optional filters with keys:
            - actors: List of agent names to filter by (e.g., ["code_agent", "supervisor", "communication_agent"])
            - start_time: ISO timestamp string for earliest conversation time (e.g., "2026-02-12T00:00:00")
            - end_time: ISO timestamp string for latest conversation time (e.g., "2026-02-13T23:59:59")

    Example: retrieve_relevant_chunks("Python script debugging", 5, {"actors": ["code_agent"], "start_time": None, "end_time": None})
    """
    try:
        rag = get_rag_instance()
        results = await asyncio.to_thread(rag.retrieve_chunks, query, conditions, top_k)

        return results
    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {e}")
        return f"Error retrieving relevant chunks: {e}"


# --- Standing instructions: rules from SIR, shown to every agent on every message ----


async def get_instructions_text() -> str:
    """The active standing instructions as prompt lines ("- [id] text"); "" when there
    are none or memory is off (MEMORY_RECALL)."""
    if not MEMORY_RECALL:
        return ""
    try:
        rows = await list_instructions()
    except Exception as e:
        logger.error(f"Loading standing instructions failed: {e}")
        return ""
    return "\n".join(f"- [{row_id}] {text}" for row_id, text, _created in rows)


@supervisor_server.tool()
async def remember_instruction(instruction: str):
    """
    Save a standing instruction from SIR: how something should be done from now on
    ("from now on...", "always...", "never...", "hereafter...", "next time...").
    Write it as one clear, self-contained sentence for the assistant, e.g.
    "Before sending any email, show SIR the recipient and full content and wait for approval."
    It is shown to you and every worker on every message until SIR drops it.
    For facts about people, projects or organizations use add_information_to_knowledge_graph.

    Args:
        instruction: The rule, as one sentence.
    """
    if not instruction or not instruction.strip():
        return "Nothing saved: the instruction was empty."
    try:
        instruction_id = await add_instruction(instruction)
        return f"Saved standing instruction [{instruction_id}]: {instruction.strip()}"
    except Exception as e:
        logger.error(f"Saving standing instruction failed: {e}")
        return f"Could not save the instruction: {e}"


@supervisor_server.tool()
async def forget_instruction(instruction_id: int):
    """
    Drop a standing instruction SIR no longer wants, by the [id] shown in the list of
    standing instructions. To change one, drop it and save the new wording.

    Args:
        instruction_id: The number shown in brackets, e.g. 3 for "[3] ...".
    """
    try:
        if await remove_instruction(instruction_id):
            return f"Standing instruction [{instruction_id}] dropped."
        return f"No active standing instruction [{instruction_id}]."
    except Exception as e:
        logger.error(f"Dropping standing instruction failed: {e}")
        return f"Could not drop the instruction: {e}"
