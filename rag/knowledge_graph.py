import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import kuzu
import re
from datetime import datetime
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import sys
import pandas
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from typing import List, Literal

from config.prompts import (
    KNOWLEDGE_GRAPH_EXTRACTION_PROMPT,
    KNOWLEDGE_GRAPH_VALIDATION_PROMPT,
)
from config.settings import EMBEDDING_BGE_MODEL_PATH, KNOWLEDGE_GRAPH_DB
from utils.helper import first_line, load_sentence_model, setup_logger

logger = setup_logger(__name__)

EntityType = Literal[
    "Person", "Project", "Organization", "Tool", "Concept", "Event", "Resource"
]
ALLOWED_ENTITY_TYPES = (
    "Person", "Project", "Organization", "Tool", "Concept", "Event", "Resource"
)

# Free-form types the extraction model sometimes invents ("Educational Institution",
# "Career Role", "Platform"), mapped onto the allowed ones by whole-word keywords, in
# this order; anything else becomes Concept.
_TYPE_KEYWORDS = (
    ("Person", ("person", "people", "contact", "friend", "colleague", "user")),
    ("Organization", ("organization", "organisation", "institution", "company", "club",
                      "society", "community", "university", "college", "school", "team",
                      "employer", "startup", "group")),
    ("Event", ("event", "meeting", "conference", "hackathon", "deadline", "interview")),
    ("Concept", ("concept", "idea", "approach", "method", "skill", "field", "topic",
                 "preference", "goal", "interest", "role", "requirement", "area", "scope",
                 "cohort", "focus", "request", "activity", "preparation")),
    ("Project", ("project", "research", "product")),
    ("Tool", ("tool", "technology", "platform", "framework", "library", "database",
              "software", "language", "model", "service")),
    ("Resource", ("resource", "data", "dataset", "document", "file", "link", "paper",
                  "book", "course")),
)


def normalize_entity_type(entity_type) -> str:
    """One of the seven allowed entity types."""
    text = str(entity_type or "").strip()
    for allowed in ALLOWED_ENTITY_TYPES:
        if text.lower() == allowed.lower():
            return allowed
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).lower()  # "LearningPreference"
    for allowed, keywords in _TYPE_KEYWORDS:
        if any(re.search(rf"\b{keyword}s?\b", words) for keyword in keywords):
            return allowed
    return "Concept"


def normalize_relation_type(relation_type) -> str:
    """UPPER_SNAKE_CASE ("worksWith", "works with" -> "WORKS_WITH")."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(relation_type or "").strip())
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()
    return text or "RELATED_TO"


# The only relationship names stored. The extraction and validation prompts list them
# ({relation_types}); anything else the model writes is mapped onto them below.
RELATION_TYPES = {
    "KNOWS": "a person knows or is a friend of another person",
    "WORKS_WITH": "a person collaborates with another person",
    "REPORTS_TO": "a person reports to their manager",
    "MEMBER_OF": "a person belongs to an organization, club, team or group",
    "WORKS_AT": "a person currently works at an organization",
    "WORKED_AT": "a person worked at an organization before",
    "INTERNED_AT": "a person did an internship at an organization",
    "STUDIES_AT": "a person studies at a school or university",
    "STUDIES": "a person studies a subject",
    "WORKS_ON": "a person is working on a project or activity now",
    "WORKED_ON": "a person worked on a project or activity before",
    "BUILT": "a person built or created a project or system",
    "USES": "a person or project uses a tool, technology or platform",
    "EXPERIENCED_IN": "a person has experience or skills in a field",
    "INTERESTED_IN": "a person is interested in or follows a topic, field or option",
    "PREFERS": "a person prefers or values a way of working, style or choice",
    "PREPARES_FOR": "a person works towards a goal (a role, exam, placement, internship, skill)",
    "PART_OF": "something is a part, topic or component of something bigger",
    "REQUIRES": "something needs or depends on something else",
    "LOCATED_IN": "something or someone is in a place",
    "ATTENDS": "a person attends or takes part in an event",
    "RELATED_TO": "any other link (only when none of the above fits)",
}

RELATION_TYPES_TEXT = "\n".join(f"- {name}: {meaning}" for name, meaning in RELATION_TYPES.items())

# Other names the model uses, as (stored name, reverse direction). "A INVOLVES B" is
# stored as "B PART_OF A".
_RELATION_SYNONYMS = {
    "FRIEND_OF": ("KNOWS", False), "FRIENDS_WITH": ("KNOWS", False), "MET": ("KNOWS", False),
    "COLLABORATES_WITH": ("WORKS_WITH", False), "COLLEAGUE_OF": ("WORKS_WITH", False),
    "MANAGED_BY": ("REPORTS_TO", False), "MANAGES": ("REPORTS_TO", True),
    "BELONGS_TO": ("MEMBER_OF", False), "JOINED": ("MEMBER_OF", False),
    "HAS_MEMBER": ("MEMBER_OF", True),
    "EMPLOYED_AT": ("WORKS_AT", False), "EMPLOYED_BY": ("WORKS_AT", False),
    "WORKS_FOR": ("WORKS_AT", False), "WORKED_FOR": ("WORKED_AT", False),
    "INTERNS_AT": ("INTERNED_AT", False), "INTERN_AT": ("INTERNED_AT", False),
    "STUDIED_AT": ("STUDIES_AT", False), "ENROLLED_AT": ("STUDIES_AT", False),
    "MAJORS_IN": ("STUDIES", False),
    "WORKING_ON": ("WORKS_ON", False), "ENGAGES_IN": ("WORKS_ON", False),
    "CONDUCTS": ("WORKS_ON", False), "FOCUSES_ON": ("WORKS_ON", False),
    "FOCUSES_ON_BUILDING": ("WORKS_ON", False),
    "CONDUCTED": ("WORKED_ON", False), "EXPERIMENTED_WITH": ("WORKED_ON", False),
    "PROCESSED": ("WORKED_ON", False), "COMPLETED": ("WORKED_ON", False),
    "BUILDS": ("BUILT", False), "CREATED": ("BUILT", False), "DEVELOPED": ("BUILT", False),
    "USED": ("USES", False), "UTILIZES": ("USES", False),
    "WORKS_IN": ("EXPERIENCED_IN", False), "SKILLED_IN": ("EXPERIENCED_IN", False),
    "INTERESTED_IN_BUILDING": ("INTERESTED_IN", False), "FOLLOWS": ("INTERESTED_IN", False),
    "CONSIDERS": ("INTERESTED_IN", False), "ALSO_CONSIDERS": ("INTERESTED_IN", False),
    "REQUESTS": ("INTERESTED_IN", False),
    "STRONGLY_PREFERS": ("PREFERS", False), "VALUES": ("PREFERS", False), "LIKES": ("PREFERS", False),
    "PREPARING_FOR": ("PREPARES_FOR", False), "PURSUES": ("PREPARES_FOR", False),
    "PURSUING": ("PREPARES_FOR", False), "AIMS_FOR": ("PREPARES_FOR", False),
    "WANTS_TO_STRENGTHEN": ("PREPARES_FOR", False),
    "INVOLVES": ("PART_OF", True), "INCLUDES": ("PART_OF", True), "COVERS": ("PART_OF", True),
    "HAS_PART": ("PART_OF", True), "CONTAINS": ("PART_OF", True),
    "HAS_GEOGRAPHIC_SCOPE": ("PART_OF", True),
    "NEEDS": ("REQUIRES", False), "DEPENDS_ON": ("REQUIRES", False),
    "LIVES_IN": ("LOCATED_IN", False), "BASED_IN": ("LOCATED_IN", False),
    "ATTENDED": ("ATTENDS", False), "PARTICIPATES_IN": ("ATTENDS", False),
    "PARTICIPATED_IN": ("ATTENDS", False),
    "USES_TOOL": ("USES", False), "USES_TECHNOLOGY": ("USES", False),
    "TARGETS_ROLE": ("PREPARES_FOR", False), "CAREER_GOAL": ("PREPARES_FOR", False),
    "CAREER_FOCUS": ("PREPARES_FOR", False), "CONSIDERS_ROLE": ("INTERESTED_IN", False),
    "INVOLVED_WITH": ("WORKS_ON", False), "INVOLVED_IN": ("WORKS_ON", False),
    "RESEARCHED": ("WORKED_ON", False),
}

# For names in neither list: the first rule with a word of the name starting with one of
# its prefixes wins (whole words, so "FOCUSES" is not read as "USES").
_RELATION_KEYWORDS = (
    ("PREPARES_FOR", ("GOAL", "TARGET", "CAREER", "ASPIR", "PURSU", "PREPAR", "AIM")),
    ("INTERESTED_IN", ("CONSIDER", "INTEREST", "FOLLOW", "CURIOUS")),
    ("PREFERS", ("PREFER", "VALUE", "LIKE", "ENJOY", "FAVO")),
    ("MEMBER_OF", ("MEMBER", "BELONG", "JOIN")),
    ("INTERNED_AT", ("INTERN",)),
    ("STUDIES", ("STUD", "LEARN", "MAJOR")),
    ("BUILT", ("BUILT", "BUILD", "CREAT", "DEVELOP", "MADE")),
    ("WORKED_ON", ("RESEARCHED", "WORKED", "COMPLETED", "EXPERIMENTED", "CONTRIBUTED")),
    ("WORKS_ON", ("RESEARCH", "WORK", "FOCUS", "INVOLV", "CONTRIBUT", "LEAD")),
    ("USES", ("USE", "USING", "UTILIZ")),
    ("KNOWS", ("FRIEND", "KNOW")),
    ("LOCATED_IN", ("LOCAT", "LIVE", "BASED")),
    ("REQUIRES", ("REQUIR", "NEED", "DEPEND")),
    ("ATTENDS", ("ATTEND", "PARTICIPAT")),
)


def _relation_from_keywords(name: str):
    words = [w for w in name.split("_") if w]
    for stored, prefixes in _RELATION_KEYWORDS:
        if any(word.startswith(prefix) for word in words for prefix in prefixes):
            return stored
    return None


def canonical_relation(relation_type, source_type=None, target_type=None) -> tuple:
    """(stored name, reverse) for a relationship the model wrote, using the entity types
    where the meaning depends on them: "works with" a tool is USES, and a project that
    "belongs to" something is PART_OF it, not a member."""
    name = normalize_relation_type(relation_type)
    reverse = False
    if name in _RELATION_SYNONYMS:
        name, reverse = _RELATION_SYNONYMS[name]
    elif name not in RELATION_TYPES:
        # Unknown name: a link to a tool is using it; otherwise the name's words decide.
        name = "USES" if target_type == "Tool" else (_relation_from_keywords(name) or "RELATED_TO")
    if reverse:
        source_type, target_type = target_type, source_type
    if name in ("WORKS_WITH", "KNOWS") and target_type == "Tool":
        name = "USES"
    elif name == "MEMBER_OF" and source_type not in (None, "Person"):
        name = "PART_OF"
    elif name == "WORKS_ON" and target_type == "Organization":
        name = "MEMBER_OF"  # "involved with" a club or society
    elif name == "WORKED_ON" and target_type == "Organization":
        name = "WORKED_AT"
    return name, reverse


def normalize_candidates(candidates_json: dict) -> dict:
    """Allowed types and stored relationship names in extraction output (in place)."""
    candidates = (candidates_json or {}).get("candidates") or {}
    types = {}
    for entity in candidates.get("entities") or []:
        entity["type"] = normalize_entity_type(entity.get("type"))
        types[str(entity.get("id", "")).lower()] = entity["type"]
    for relationship in candidates.get("relationships") or []:
        name, reverse = canonical_relation(
            relationship.get("relation_type"),
            types.get(str(relationship.get("source", "")).lower()),
            types.get(str(relationship.get("target", "")).lower()),
        )
        relationship["relation_type"] = name
        if reverse:
            relationship["source"], relationship["target"] = (
                relationship.get("target"),
                relationship.get("source"),
            )
    return candidates_json


# How look-alikes are found for each new entity: an exact name match (any
# capitalization) always counts; otherwise at most LOOKALIKE_TOP_K stored entities whose
# embedding similarity is at least LOOKALIKE_MIN_SCORE. bge-small rates even different
# things in one field highly (TensorFlow vs Keras ~0.95), so a low cut-off floods the
# validator with candidates.
LOOKALIKE_MIN_SCORE = 0.8
LOOKALIKE_TOP_K = 5


def _name_key(name) -> str:
    """What identifies an entity name: letters and digits only, any case, so
    "Customer-Support", "customer support" and "Customer Support" are one entity."""
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def _combine_text(old: str, new: str) -> str:
    """Two descriptions of the same entity as one, without repeating either."""
    old, new = (old or "").strip(), (new or "").strip()
    if not old or old.lower() in new.lower():
        return new
    if not new or new.lower() in old.lower():
        return old
    return f"{old.rstrip('.')}. {new}"


class KnowledgeEntity(BaseModel):
    id: str = Field(..., description="Unique name of the entity (e.g., 'DeepShield')")
    type: EntityType = Field(..., description="The category of the entity")
    search_keywords: List[str] = Field(
        ..., description="3-5 search keywords for vector retrieval"
    )
    description: str = Field(..., description="A 1-sentence summary of what this is")


class KnowledgeRelationship(BaseModel):
    source: str = Field(..., description="The id of the starting node")
    target: str = Field(..., description="The id of the ending node")
    relation_type: str = Field(
        ..., description="Relationship name (e.g., 'USES', 'DEVELOPED_AT')"
    )


class KnowledgeGraph:
    def __init__(self, db_path: str = KNOWLEDGE_GRAPH_DB):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = kuzu.Database(str(self.path))
        self.conn = kuzu.Connection(self.db)
        self._model = None
        self._create_generic_schema()

    @property
    def model(self):
        if self._model is None:
            try:
                self._model = load_sentence_model(
                    EMBEDDING_BGE_MODEL_PATH, "BAAI/bge-small-en-v1.5"
                )
                logger.info("SentenceTransformer model loaded successfully.")
            except Exception as e:
                logger.error(
                    f"Could not load the knowledge-graph embedding model: {first_line(e)}"
                )
                raise
        return self._model

    def _create_generic_schema(self):
        try:
            try:
                self.conn.execute("MATCH (n:Entity) RETURN n LIMIT 1")
                logger.info("Entity table already exists.")
            except Exception:
                self.conn.execute("""
                    CREATE NODE TABLE Entity(
                        id STRING, 
                        type STRING, 
                        search_keywords STRING, 
                        description STRING, 
                        embedding FLOAT[384], 
                        PRIMARY KEY(id)
                    )
                """)
                logger.info("Entity table created successfully.")
            try:
                self.conn.execute(
                    "CALL CREATE_VECTOR_INDEX('Entity', 'Entity_embedding_idx', 'embedding', metric := 'COSINE');"
                )
                logger.info("HNSW vector index created successfully.")
            except Exception as e:
                logger.info(f"Error creating HNSW index (might already exist): {e}")

            try:
                self.conn.execute("MATCH ()-[r:RELATED_TO]->() RETURN r LIMIT 1")
                logger.info("RELATED_TO table already exists.")
            except Exception:
                self.conn.execute("""
                    CREATE REL TABLE RELATED_TO(
                        FROM Entity TO Entity, 
                        relation_type STRING
                    )
                """)
                logger.info("RELATED_TO table created successfully.")

            self._ensure_freshness_columns()
        except Exception as e:
            logger.info(f"Error creating schema: {e}")

    # When each fact was last stored and where it came from, so newer facts can replace
    # outdated ones and recall can show how old a fact is. Added to existing graphs too.
    FRESHNESS_COLUMNS = (
        ("Entity", "updated_at"),
        ("Entity", "learned_from"),
        ("RELATED_TO", "updated_at"),
    )

    def _ensure_freshness_columns(self):
        for table, column in self.FRESHNESS_COLUMNS:
            try:
                self.conn.execute(f"ALTER TABLE {table} ADD {column} STRING DEFAULT ''")
                logger.info(f"Added {table}.{column} to the knowledge graph.")
            except Exception:
                pass  # the column already exists

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    def count_entities(self) -> int:
        result = self.conn.execute("MATCH (n:Entity) RETURN count(n)")
        return result.get_next()[0] if result.has_next() else 0

    def _resolve_id(self, name: str) -> str:
        """The stored id for a name, matched regardless of case and punctuation (the name
        itself if it is not stored yet), so "AI systems" and "AI Systems", or
        "Customer-Support" and "Customer Support", are one entity."""
        found = self.conn.execute(
            "MATCH (n:Entity) WHERE lower(n.id) = lower($id) RETURN n.id ORDER BY n.id LIMIT 1",
            parameters={"id": str(name)},
        )
        if found.has_next():
            return found.get_next()[0]
        key = _name_key(name)
        if key:
            stored = self.conn.execute("MATCH (n:Entity) RETURN n.id ORDER BY n.id")
            while stored.has_next():
                stored_id = stored.get_next()[0]
                if _name_key(stored_id) == key:
                    return stored_id
        return name

    def _stored_entity(self, node_id: str):
        rows = (
            self.conn.execute(
                """
                MATCH (n:Entity {id: $id})
                RETURN n.id AS id, n.type AS type, n.description AS description,
                    n.search_keywords AS search_keywords, n.learned_from AS learned_from
                """,
                parameters={"id": node_id},
            )
            .get_as_df()
            .to_dict("records")
        )
        return rows[0] if rows else None

    def warm_up(self):
        """Load the embedding model now (startup), not on the first lookup."""
        return self.model is not None

    @staticmethod
    def _entity_text(node_id, node_type, description, search_keywords) -> str:
        """The text an entity's embedding is computed from. Stored entities and the
        candidates compared against them use the same template, so their similarity
        reflects the facts rather than two different phrasings."""
        if isinstance(search_keywords, list):
            search_keywords = ", ".join(search_keywords)
        return f"{node_id} is a {node_type}. {description or ''} Keywords: {search_keywords or ''}"

    def entities_named_in(self, text: str) -> list:
        """Ids of entities whose name appears in the text as whole words (any case)."""
        result = self.conn.execute("MATCH (n:Entity) RETURN n.id")
        names = []
        while result.has_next():
            name = result.get_next()[0]
            if name and len(name) >= 2 and re.search(
                rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE
            ):
                names.append(name)
        return names

    def entity_facts(self, ids: list) -> list:
        """Stored entities by id, as dicts (id, type, description, updated_at)."""
        if not ids:
            return []
        return (
            self.conn.execute(
                """
                MATCH (n:Entity) WHERE n.id IN $ids
                RETURN n.id AS id, n.type AS type, n.description AS description,
                    n.updated_at AS updated_at
                """,
                parameters={"ids": list(ids)},
            )
            .get_as_df()
            .to_dict("records")
        )

    def relationships_of(self, ids: list, limit: int = 20) -> list:
        """Relationships touching any of the ids, newest first, as dicts
        (source, relation_type, target, updated_at)."""
        if not ids:
            return []
        return (
            self.conn.execute(
                """
                MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
                WHERE a.id IN $ids OR b.id IN $ids
                RETURN a.id AS source, r.relation_type AS relation_type, b.id AS target,
                    r.updated_at AS updated_at
                ORDER BY updated_at DESC
                LIMIT $limit
                """,
                parameters={"ids": list(ids), "limit": limit},
            )
            .get_as_df()
            .to_dict("records")
        )

    def profile(self, name: str, limit: int = 12) -> dict:
        """The user's own entity and the people, projects and organizations directly
        connected to it, newest relationship first.

        Returns {"entity": dict or None, "connections": [dict]} where each connection has
        relation_type, outgoing (user -> other), id, type, description, updated_at.
        """
        found = (
            self.conn.execute(
                """
                MATCH (u:Entity) WHERE lower(u.id) = lower($name)
                RETURN u.id AS id, u.type AS type, u.description AS description,
                    u.updated_at AS updated_at
                """,
                parameters={"name": name},
            )
            .get_as_df()
            .to_dict("records")
        )
        if not found:
            return {"entity": None, "connections": []}
        user = found[0]

        connections = []
        for outgoing, pattern in (
            (True, "(u:Entity {id: $id})-[r:RELATED_TO]->(m:Entity)"),
            (False, "(m:Entity)-[r:RELATED_TO]->(u:Entity {id: $id})"),
        ):
            rows = (
                self.conn.execute(
                    f"""
                    MATCH {pattern}
                    WHERE m.id <> $id
                    RETURN r.relation_type AS relation_type, r.updated_at AS rel_updated_at,
                        m.id AS id, m.type AS type, m.description AS description,
                        m.updated_at AS updated_at
                    """,
                    parameters={"id": user["id"]},
                )
                .get_as_df()
                .to_dict("records")
            )
            connections += [{**row, "outgoing": outgoing} for row in rows]

        connections.sort(
            key=lambda c: c.get("rel_updated_at") or c.get("updated_at") or "",
            reverse=True,
        )
        return {"entity": user, "connections": connections[:limit]}

    def execute_query(self, query: str):
        try:
            return self.conn.execute(query)
        except Exception as e:
            logger.info(f"Error executing query: {e}")
            return None

    def clear_database(self):
        try:
            self.execute_query("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared successfully.")
        except Exception as e:
            logger.info(f"Error clearing database: {e}")

    def _compute_embedding(self, text: str, is_query: bool = False):
        try:
            if is_query:
                text = (
                    f"Represent this sentence for searching relevant passages: {text}"
                )
            embedding = self.model.encode(text, normalize_embeddings=True)

            if hasattr(embedding, "tolist"):
                return embedding.tolist()
            return list(embedding)
        except Exception as e:
            logger.info(f"Error computing embedding: {e}")
            return None

    def close(self):
        self.conn.close()

    def add_entity(
        self,
        node_id: str,
        node_type: str,
        search_keywords: str,
        description: str,
        learned_from: str = "",
    ):
        """Create or replace an entity, keeping its relationships.

        kuzu cannot SET the vector-indexed embedding column, so an update deletes the
        node and inserts it again. Its relationships are read first and re-created
        after, in one transaction, so updating a fact no longer drops its connections.
        """
        try:
            node_type = normalize_entity_type(node_type)
            stored_id = self._resolve_id(node_id)
            if stored_id != node_id:
                # The same entity under another capitalization: update it, keeping what
                # it already says, instead of adding a second node.
                stored = self._stored_entity(stored_id) or {}
                description = _combine_text(stored.get("description"), description)
                search_keywords = ", ".join(
                    dict.fromkeys(
                        k.strip()
                        for k in f"{stored.get('search_keywords') or ''},{search_keywords or ''}".split(",")
                        if k.strip()
                    )
                )
                learned_from = learned_from or stored.get("learned_from") or ""
                node_id = stored_id
            text = self._entity_text(node_id, node_type, description, search_keywords)
            embedding = self._compute_embedding(text)
            if embedding is None:
                logger.error(f"No embedding for '{node_id}'; left the graph unchanged.")
                return

            outgoing = self.conn.execute(
                """
                MATCH (n:Entity {id: $id})-[r:RELATED_TO]->(m:Entity)
                RETURN m.id AS other, r.relation_type AS relation_type, r.updated_at AS updated_at
                """,
                parameters={"id": node_id},
            ).get_as_df()
            incoming = self.conn.execute(
                """
                MATCH (m:Entity)-[r:RELATED_TO]->(n:Entity {id: $id})
                WHERE m.id <> $id
                RETURN m.id AS other, r.relation_type AS relation_type, r.updated_at AS updated_at
                """,
                parameters={"id": node_id},
            ).get_as_df()

            self.conn.execute("BEGIN TRANSACTION")
            try:
                self.conn.execute(
                    "MATCH (n:Entity {id: $id}) DETACH DELETE n",
                    parameters={"id": node_id},
                )
                self.conn.execute(
                    """
                    CREATE (n:Entity {
                        id: $id,
                        type: $type,
                        search_keywords: $search_keywords,
                        description: $description,
                        embedding: $embedding,
                        updated_at: $updated_at,
                        learned_from: $learned_from
                    })
                    """,
                    parameters={
                        "id": node_id,
                        "type": node_type,
                        "search_keywords": search_keywords,
                        "description": description,
                        "embedding": embedding,
                        "updated_at": self._now(),
                        "learned_from": learned_from or "",
                    },
                )
                for rows, outward in ((outgoing, True), (incoming, False)):
                    for row in rows.to_dict("records"):
                        other = row["other"]
                        self._create_relationship(
                            node_id if outward else other,
                            other if outward else node_id,
                            row["relation_type"],
                            row["updated_at"] or "",
                        )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            logger.info(
                f"✅ Node '{node_id}' stored with aligned embedding "
                f"({len(outgoing) + len(incoming)} relationships kept)."
            )
        except Exception as e:
            logger.error(f"Error adding node: {e}")

    def _create_relationship(
        self, source: str, target: str, relation_type: str, updated_at: str
    ):
        self.conn.execute(
            """
            MATCH (a:Entity {id: $source}), (b:Entity {id: $target})
            CREATE (a)-[:RELATED_TO {relation_type: $relation_type, updated_at: $updated_at}]->(b)
            """,
            parameters={
                "source": source,
                "target": target,
                "relation_type": relation_type,
                "updated_at": updated_at,
            },
        )

    def add_relationship(self, source: str, target: str, relation_type: str):
        """Connect two entities (found in any capitalization). Storing the same
        relationship again refreshes its timestamp instead of adding a duplicate edge."""
        try:
            source, target = self._resolve_id(source), self._resolve_id(target)
            name, reverse = canonical_relation(
                relation_type, self._entity_type(source), self._entity_type(target)
            )
            if reverse:
                source, target = target, source
            self._upsert_relationship(source, target, name)
        except Exception as e:
            logger.info(f"Error adding relationship: {e}")
            return

    def _entity_type(self, node_id: str):
        stored = self._stored_entity(node_id)
        return stored["type"] if stored else None

    def _upsert_relationship(self, source: str, target: str, relation_type: str):
        """add_relationship for exact ids and an already-normalized name."""
        existing = self.conn.execute(
            """
            MATCH (a:Entity {id: $source})-[r:RELATED_TO]->(b:Entity {id: $target})
            WHERE r.relation_type = $relation_type
            SET r.updated_at = $updated_at
            RETURN count(r)
            """,
            parameters={
                "source": source,
                "target": target,
                "relation_type": relation_type,
                "updated_at": self._now(),
            },
        )
        if existing.has_next() and existing.get_next()[0]:
            return
        self._create_relationship(source, target, relation_type, self._now())

    # sometimes overriding is better than addition failure so need to work on that
    # def modify_entity(self, node_id: str, updates: dict):
    #     try:
    #         if not updates:
    #             return
    #         query = f"MATCH (n:Entity) WHERE n.id = '{node_id}' RETURN n"
    #         result = self.execute_query(query)
    #         if not result or not result.has_next():
    #             logger.info(f"Node '{node_id}' not found for update.")
    #             return
    #         type = ""
    #         search_keywords = ""
    #         if "type" not in updates:
    #             query = (
    #                 f"MATCH (n:Entity) WHERE n.id = '{node_id}' RETURN n.type AS type"
    #             )
    #             type = self.execute_query(query)
    #         if "search_keywords" not in updates:
    #             query = f"""
    #             MATCH (n:Entity) WHERE n.id = '{node_id}' RETURN n.search_keywords AS search_keywords
    #             """
    #             search_keywords = self.execute_query(query)

    #         text = f"Entity: {node_id} | Type: {updates.get('type', type)} | Keywords: {updates.get('search_keywords', search_keywords)}"
    #         new_embedding = self.model.encode(text)
    #         updates["embedding"] = new_embedding

    #         set_parts = []
    #         for key, value in updates.items():
    #             if isinstance(value, str):
    #                 safe_val = value.replace("'", "\\'")
    #                 set_parts.append(f"n.{key} = '{safe_val}'")
    #             else:
    #                 set_parts.append(f"n.{key} = {value}")

    #         set_clauses = ", ".join(set_parts)

    #         query = f"""
    #         MATCH (n:Entity)
    #         WHERE n.id = '{node_id}'
    #         SET {set_clauses}
    #         """

    #         self.execute_query(query)
    #         logger.info(f"✅ Node '{node_id}' updated successfully in KuzuDB.")

    #     except Exception as e:
    #         logger.error(f"❌ Error modifying node '{node_id}': {e}")

    def modify_relationship(self, source: str, target: str, relation_type: str):
        try:
            self.conn.execute(
                """
                MATCH (a:Entity {id: $source})-[r:RELATED_TO]->(b:Entity {id: $target})
                SET r.relation_type = $relation_type, r.updated_at = $updated_at
                """,
                parameters={
                    "source": self._resolve_id(source),
                    "target": self._resolve_id(target),
                    "relation_type": canonical_relation(
                        relation_type,
                        self._entity_type(self._resolve_id(source)),
                        self._entity_type(self._resolve_id(target)),
                    )[0],
                    "updated_at": self._now(),
                },
            )
            logger.info(
                f"Relationship from '{source}' to '{target}' updated successfully."
            )
        except Exception as e:
            logger.info(f"Error modifying relationship: {e}")
            return

    def merge_entities(self, keep_id: str, drop_id: str) -> bool:
        """Fold drop_id into keep_id: its relationships move over (without duplicates),
        the two descriptions and keyword lists are combined, then drop_id is deleted."""
        keep, drop = self._stored_entity(keep_id), self._stored_entity(drop_id)
        if not keep or not drop or keep["id"] == drop["id"]:
            return False
        for rel in self.relationships_of([drop["id"]], limit=10_000):
            source = keep["id"] if rel["source"] == drop["id"] else rel["source"]
            target = keep["id"] if rel["target"] == drop["id"] else rel["target"]
            if source != target:
                self._upsert_relationship(source, target, rel["relation_type"])
        self.conn.execute(
            "MATCH (n:Entity {id: $id}) DETACH DELETE n", parameters={"id": drop["id"]}
        )
        keywords = ", ".join(
            dict.fromkeys(
                k.strip()
                for k in f"{keep.get('search_keywords') or ''},{drop.get('search_keywords') or ''}".split(",")
                if k.strip()
            )
        )
        self.add_entity(
            keep["id"],
            keep["type"],
            keywords,
            _combine_text(keep.get("description"), drop.get("description")),
            learned_from=keep.get("learned_from") or drop.get("learned_from") or "",
        )
        logger.info(f"Merged '{drop['id']}' into '{keep['id']}'.")
        return True

    def tidy(self, merges=(), relation_renames=None) -> dict:
        """Clean up a graph built before the quality guards existed.

        - Entities whose names differ only in case or punctuation are merged (the one
          with more relationships is kept).
        - merges: extra (keep, drop) pairs to merge, for duplicates under other names.
        - Types outside the seven allowed ones are mapped onto them (re-embedded).
        - Relationship names are normalized; relation_renames maps further names (e.g.
          a typo) to the right one. Duplicate edges are removed.
        Returns counts of what changed.
        """
        report = {"merged": [], "retyped": 0, "renamed": 0, "duplicate_edges_removed": 0}

        ids = [r[0] for r in self.conn.execute("MATCH (n:Entity) RETURN n.id").get_as_df().values]
        degree = {
            row["id"]: row["degree"]
            for row in self.conn.execute(
                "MATCH (n:Entity) OPTIONAL MATCH (n)-[r:RELATED_TO]-() RETURN n.id AS id, count(r) AS degree"
            ).get_as_df().to_dict("records")
        }
        by_name = {}
        for node_id in ids:
            by_name.setdefault(_name_key(node_id) or node_id, []).append(node_id)
        pairs = []
        for variants in by_name.values():
            if len(variants) > 1:
                variants.sort(key=lambda v: (-degree.get(v, 0), v))
                pairs += [(variants[0], other) for other in variants[1:]]
        for keep, drop in [*pairs, *merges]:
            if self.merge_entities(keep, drop):
                report["merged"].append((keep, drop))

        for row in self.conn.execute(
            "MATCH (n:Entity) RETURN n.id AS id, n.type AS type, n.description AS description, "
            "n.search_keywords AS search_keywords, n.learned_from AS learned_from"
        ).get_as_df().to_dict("records"):
            if row["type"] not in ALLOWED_ENTITY_TYPES:
                self.add_entity(row["id"], row["type"], row["search_keywords"] or "",
                                row["description"] or "", learned_from=row["learned_from"] or "")
                report["retyped"] += 1

        renames = {k.upper(): normalize_relation_type(v) for k, v in (relation_renames or {}).items()}
        edges = self.conn.execute(
            "MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) "
            "RETURN a.id AS source, a.type AS source_type, r.relation_type AS relation_type, "
            "b.id AS target, b.type AS target_type, r.updated_at AS updated_at"
        ).get_as_df().to_dict("records")
        wanted = {}
        for edge in edges:
            name = normalize_relation_type(edge["relation_type"])
            name, reverse = canonical_relation(
                renames.get(name, name), edge["source_type"], edge["target_type"]
            )
            source, target = edge["source"], edge["target"]
            if reverse:
                source, target = target, source
            key = (source, name, target)
            wanted[key] = max(wanted.get(key, ""), edge["updated_at"] or "")
            if name != edge["relation_type"] or reverse:
                report["renamed"] += 1
        report["duplicate_edges_removed"] = len(edges) - len(wanted)
        if report["renamed"] or report["duplicate_edges_removed"]:
            self.conn.execute("BEGIN TRANSACTION")
            try:
                self.conn.execute("MATCH ()-[r:RELATED_TO]->() DELETE r")
                for (source, name, target), updated_at in wanted.items():
                    self._create_relationship(source, target, name, updated_at)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        return report

    def delete_entity(self, node_id: str):
        try:
            query = f"""
            MATCH (n:Entity) 
            WHERE n.id = '{node_id}' 
            DETACH DELETE n
            """
            self.execute_query(query)
            logger.info(f"Entity '{node_id}' deleted successfully.")
        except Exception as e:
            logger.info(f"Error deleting entity: {e}")
            return

    def delete_relationship(self, source: str, target: str):
        try:
            query = f"""
            MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
            WHERE a.id = '{source}' AND b.id = '{target}'
            DELETE r
            """
            self.execute_query(query)
            logger.info(
                f"Relationship from '{source}' to '{target}' deleted successfully."
            )
        except Exception as e:
            logger.info(f"Error deleting relationship: {e}")
            return

    def find_similar_nodes(self, keywords: str, top_k: int = 5):
        try:
            query_embedding = self._compute_embedding(keywords)

            query = """
                CALL QUERY_VECTOR_INDEX('Entity', 'Entity_embedding_idx', $query_embedding, $top_k)
                YIELD node, distance
                WITH node AS n, (1 - distance) AS score
                WHERE score > 0.35
                RETURN
                    n.id AS id,
                    n.type AS type,
                    n.description AS description,
                    n.updated_at AS updated_at,
                    score AS base_score
                ORDER BY base_score DESC
                LIMIT $top_k
            """

            result = self.conn.execute(
                query,
                parameters={
                    "query_embedding": query_embedding,
                    "top_k": top_k,
                },
            )

            df = result.get_as_df()

            if not df.empty:
                df = df.sort_values(by="base_score", ascending=False)
                df = df.drop_duplicates(subset=["id"], keep="first")

            return df

        except Exception as e:
            logger.info(f"Error finding similar nodes: {e}")
            return []

    def find_similar_with_expansion(self, keywords: str, top_k: int = 5):
        """
        Find similar nodes + their neighbors using standard Cypher matches.
        Independent of internal index names.
        """
        query_embedding = self._compute_embedding(keywords, is_query=True)

        query = """
        /* 1. Get Hop 0 (Seeds) */
        CALL QUERY_VECTOR_INDEX('Entity', 'Entity_embedding_idx', $query_embedding, $top_k)
        YIELD node, distance
        WITH node AS n, (1 - distance) AS seed_score
        WHERE seed_score > 0.35
        RETURN 
            n.id AS id, 
            n.type AS type, 
            n.description AS description, 
            n.updated_at AS updated_at,
            0 AS hops, 
            seed_score AS base_score,
            'DIRECT_HIT' AS Relations
        ORDER BY base_score DESC
        LIMIT $top_k

        UNION ALL

        /* 2. Get Hop 1 (High-Confidence Neighbors) */
        CALL QUERY_VECTOR_INDEX('Entity', 'Entity_embedding_idx', $query_embedding, $top_k)
        YIELD node, distance
        WITH node AS n, (1 - distance) AS score
        MATCH (n)-[r]-(neighbor)
        WHERE neighbor.embedding IS NOT NULL
        WITH n, r, neighbor, score, array_cosine_similarity(neighbor.embedding, $query_embedding) AS score2
        WITH n, r, neighbor, (0.9 * score + 0.1 * score2) AS weighted_score
        WHERE weighted_score > 0.25
        RETURN 
            neighbor.id AS id, 
            neighbor.type AS type, 
            neighbor.description AS description, 
            neighbor.updated_at AS updated_at,
            1 AS hops, 
            weighted_score AS base_score,
            (n.id + ' ' + r.relation_type + ' ' + neighbor.id) AS Relations
        """

        result = self.conn.execute(
            query,
            parameters={
                "query_embedding": query_embedding,
                "top_k": top_k,
            },
        )

        df = result.get_as_df()

        if not df.empty:
            df = df.sort_values(by="base_score", ascending=False)
            df = df.drop_duplicates(subset=["id"], keep="first")

        return df

    def search_similar_node(self, entity: list) -> pandas.DataFrame:
        try:
            all_results = []
            for en in entity:
                # An entity with the same name (any case) is always a look-alike, even
                # when the new description reads differently.
                exact = self.conn.execute(
                    """
                    MATCH (n:Entity) WHERE lower(n.id) = lower($id)
                    OPTIONAL MATCH (n)-[r:RELATED_TO]-(m:Entity)
                    WITH n, collect(r.relation_type + ' with ' + m.id) AS connections
                    RETURN n.id AS id, n.type AS type, n.description AS description,
                        n.updated_at AS updated_at,
                        connections AS relations, 1.0 AS similarity_score
                    """,
                    parameters={"id": str(en.get("id", ""))},
                )
                all_results.append(exact.get_as_df())

                text = self._entity_text(
                    en.get("id", ""),
                    en.get("type", ""),
                    en.get("description", ""),
                    en.get("search_keywords", []),
                )
                query_embedding = self._compute_embedding(text)

                query = """
                CALL QUERY_VECTOR_INDEX('Entity', 'Entity_embedding_idx', $query_embedding, $top_k)
                YIELD node, distance
                WITH node AS n, (1 - distance) AS score
                WHERE score >= $min_score
                OPTIONAL MATCH (n)-[r:RELATED_TO]-(m:Entity)
                WITH n, score, collect(r.relation_type + ' with ' + m.id) AS connections
                RETURN n.id AS id, n.type AS type, n.description AS description, 
                    n.updated_at AS updated_at,
                    connections AS relations, score AS similarity_score
                ORDER BY similarity_score DESC
                """
                result = self.conn.execute(
                    query,
                    parameters={
                        "query_embedding": query_embedding,
                        "top_k": LOOKALIKE_TOP_K,
                        "min_score": LOOKALIKE_MIN_SCORE,
                    },
                )
                all_results.append(result.get_as_df())

            if not all_results:
                return pandas.DataFrame()
            df = pandas.concat(all_results, ignore_index=True).drop_duplicates(
                subset=["id"]
            )
            return df
        except Exception as e:
            logger.error(f"Error searching similar nodes: {e}")
            return pandas.DataFrame()

    def preprocess_graph(self):
        try:
            query = (
                "MATCH (n:Entity) WHERE n.embedding IS NULL RETURN n.id, n.description"
            )
            result = self.execute_query(query)

            while result.has_next():
                row = result.get_next()
                node_id, description = row[0], row[1]
                embedding = self._compute_embedding(description)
                update_query = f"MATCH (n:Entity) WHERE n.id = '{node_id}' SET n.embedding = {embedding}"
                self.execute_query(update_query)

            logger.info("Graph preprocessing complete: All nodes have embeddings.")
        except Exception as e:
            logger.info(f"Error during graph preprocessing: {e}")

    def visualize(self, output_path: str = "docs/images/knowledge_graph.png"):
        try:
            nodes_res = self.conn.execute("MATCH (n:Entity) RETURN n.id, n.type")
            rels_res = self.conn.execute(
                "MATCH (a)-[r]->(b) RETURN a.id, b.id, r.relation_type"
            )

            G = nx.DiGraph()
            node_colors = []

            color_map = {
                "Project": "#2ecc71",
                "Organization": "#3498db",
                "Tool": "#f1c40f",
                "Person": "#e67e22",
            }

            while nodes_res.has_next():
                node_id, n_type = nodes_res.get_next()
                G.add_node(node_id, type=n_type)
                node_colors.append(color_map.get(n_type, "#95a5a6"))

            while rels_res.has_next():
                u, v, relation_type = rels_res.get_next()
                G.add_edge(u, v, label=relation_type)

            plt.figure(figsize=(16, 10))

            pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

            nx.draw_networkx_nodes(
                G, pos, node_size=3000, node_color=node_colors, alpha=0.9
            )
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=G.edges(),
                edge_color="#bdc3c7",
                arrowsize=20,
                connectionstyle="arc3, rad=0.1",
            )

            edge_labels = nx.get_edge_attributes(G, "label")
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

            plt.title("Knowledge Graph", fontsize=15)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"✅ Visualization saved with improved spacing: {output_path}")

        except Exception as e:
            print(f"Error during visualization: {e}")

    def generate_entity_relation(
        self, text: str, prompt=KNOWLEDGE_GRAPH_EXTRACTION_PROMPT
    ) -> json:
        """
        Use LLM to extract knowledge graph from text.
        """
        from core.llm import build_llm

        prompt = prompt.replace("{relation_types}", RELATION_TYPES_TEXT)
        prompt = f"{prompt}\nChat History:\n{text}\n"

        llm_model = build_llm()
        response = llm_model.invoke(prompt)

        try:
            raw_text = response.content
            clean_response = raw_text.replace("```json", "").replace("```", "").strip()

            graph_data = json.loads(clean_response)
            return graph_data
        except Exception as e:
            logger.info(f"Error parsing graph extraction: {e}")
            return None

    def validate_entity_relation(
        self,
        existing_knowledge: pandas.DataFrame,
        new_knowledge: json,
        prompt=KNOWLEDGE_GRAPH_VALIDATION_PROMPT,
    ) -> bool:
        """
        Use LLM to validate the extracted knowledge graph JSON against the conversation.
        """
        from core.llm import build_llm

        # A DataFrame printed as text hides most columns ("id ... similarity_score"), so the
        # validator would only see names. Give it every field of every look-alike.
        if isinstance(existing_knowledge, pandas.DataFrame):
            existing_knowledge = existing_knowledge.to_dict("records")
        prompt = prompt.replace("{relation_types}", RELATION_TYPES_TEXT)
        prompt = (
            f"{prompt}\nExisting Knowledge:\n"
            f"{json.dumps(existing_knowledge, indent=2, default=str)}\n"
            f"New Knowledge:\n{json.dumps(new_knowledge, indent=2, default=str)}\n"
        )

        llm_model = build_llm()
        response = llm_model.invoke(prompt)

        try:
            raw_text = response.content
            clean_response = raw_text.replace("```json", "").replace("```", "").strip()

            graph_data = json.loads(clean_response)
            return graph_data
        except Exception as e:
            logger.info(f"Error parsing graph extraction: {e}")
            return None
