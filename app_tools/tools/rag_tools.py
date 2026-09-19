from rag.knowledge_graph import KnowledgeGraph
from rag.episodic_rag import EpisodicRAG
import sys
from pathlib import Path
import asyncio

from app_tools.core.server_init import supervisor_server
from utils.helper import setup_logger
import logging

logger = setup_logger(__name__)

_kg_instance = None
_rag_instance = None


def get_rag_instance():
    global _rag_instance
    if _rag_instance is None:
        try:
            _rag_instance = EpisodicRAG()
        except Exception as e:
            logger.error(f"Failed to initialize EpisodicRAG: {e}")
            raise
    return _rag_instance


def get_kg_instance():
    global _kg_instance
    if _kg_instance is None:
        try:
            _kg_instance = KnowledgeGraph()
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraph: {e}")
            raise
    return _kg_instance


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
        kg = get_kg_instance()

        explicitly_request_prompt = """
            "Extract all entities and relationships from the following text as a structured Knowledge Graph. "
            "The user is explicitly providing this information for storage. "
            "Return the output strictly in the requested JSON format."

            ### ONE-SHOT EXAMPLE
            "Yadeesh: Hey, I want to add some information about my friend Raajan. He is a collaborator of mine in a college club. His email is raanjan@gmail.com"

            **Output JSON:**
            {
            "candidates": {
                "entities": [
                {
                    "id": "Raajan",
                    "type": "Person",
                    "description": "Collaborator of Yadeesh in a college club; email: raanjan@gmail.com",
                    "search_keywords": ["raanjan", "raanjan@gmail.com", "college club"]
                }
                ],
                "relationships": [
                {
                    "source": "Yadeesh",
                    "target": "Raajan",
                    "relation_type": "WORKS_WITH"
                }
                ]
            }
            }     
        """

        candidates_json = await asyncio.to_thread(
            kg.generate_entity_relation, details, explicitly_request_prompt
        )

        entities = candidates_json.get("candidates", {}).get("entities", [])

        if not entities:
            logger.info("No entities extracted from the provided details.")
            return "No entities extracted from the provided details."

        types_df = await asyncio.to_thread(kg.search_similar_node, entities)

        final_update_json = kg.validate_entity_relation(types_df, candidates_json)
        resolution = final_update_json.get("resolution", {})

        if not resolution.get("entities") and not resolution.get("relationships"):
            logger.info("No valid entities to add after validation.")
            return "No valid entities to add after validation. so this information is already exist in knowledge graph."

        for entity in resolution.get("entities", []):
            action = entity.get("action", "DISCARD").upper()

            if action == "CREATE":
                await asyncio.to_thread(
                    kg.add_entity,
                    node_id=entity["id"],
                    node_type=entity.get("type", "unknown"),
                    search_keywords=", ".join(entity.get("search_keywords", [])),
                    description=entity.get("description", ""),
                )
            elif action == "UPDATE":
                await asyncio.to_thread(
                    kg.add_entity,
                    node_id=entity["id"],
                    node_type=entity.get("type", "unknown"),
                    search_keywords=", ".join(entity.get("search_keywords", [])),
                    description=entity.get("description", ""),
                )

        for rel in resolution.get("relationships", []):
            action = rel.get("action", "DISCARD").upper()

            if action == "CREATE":
                await asyncio.to_thread(
                    kg.add_relationship,
                    source=rel["source"],
                    target=rel["target"],
                    relation_type=rel.get("relation_type", "unknown"),
                )
            elif action == "UPDATE":
                await asyncio.to_thread(
                    kg.modify_relationship,
                    source=rel["source"],
                    target=rel["target"],
                    relation_type=rel.get("relation_type", "unknown"),
                )
        await asyncio.to_thread(kg.visualize)
        return "Information added/updated successfully in the knowledge graph."
    except Exception as e:
        logger.error(f"Error adding information to knowledge graph: {e}")
        return f"Error adding information to knowledge graph: {e}"


@supervisor_server.tool()
async def retrieve_relevant_chunks(query: str, top_k: int = 5, conditions: dict = None):
    """
    USE THIS TOOL TO SEARCH CONVERSATION HISTORY WITH EPISODIC RAG ONLY THIS IS NOT KNOWLEDGE GRAPH TOOL

    Search conversation history using Episodic RAG to retrieve past interactions, decisions, and contextual information.

    Use this to recall what was discussed in previous conversations, find code snippets mentioned earlier,
    retrieve decisions made, or locate any temporal/contextual information from past interactions.

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
