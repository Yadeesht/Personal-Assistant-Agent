import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import kuzu
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
from utils.helper import setup_logger

logger = setup_logger(__name__)

EntityType = Literal[
    "Person", "Project", "Organization", "Tool", "Concept", "Event", "Resource"
]


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
                if not Path(EMBEDDING_BGE_MODEL_PATH).exists():
                    downloaded = SentenceTransformer("BAAI/bge-small-en-v1.5")
                    downloaded.save(str(EMBEDDING_BGE_MODEL_PATH))
                self._model = SentenceTransformer(EMBEDDING_BGE_MODEL_PATH)

                logger.info("SentenceTransformer model loaded successfully.")
            except Exception as e:
                logger.info(f"Error loading SentenceTransformer model: {e}")
                raise e
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
        except Exception as e:
            logger.info(f"Error creating schema: {e}")

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
        self, node_id: str, node_type: str, search_keywords: str, description: str
    ):
        try:
            text = (
                f"{node_id} is a {node_type}. {description} Keywords: {search_keywords}"
            )
            embedding = self._compute_embedding(text)

            safe_id = node_id.replace("'", "\\'")

            delete_query = f"MATCH (n:Entity {{id: '{safe_id}'}}) DETACH DELETE n"
            self.execute_query(delete_query)

            safe_desc = description.replace("'", "\\'")
            safe_keywords = search_keywords.replace("'", "\\'")

            query = f"""
            CREATE (n:Entity {{
                id: '{safe_id}',
                type: '{node_type}',
                search_keywords: '{safe_keywords}',
                description: '{safe_desc}',
                embedding: {embedding}
            }})
            """
            self.execute_query(query)
            logger.info(f"✅ Node '{node_id}' re-created with aligned embedding.")
        except Exception as e:
            logger.error(f"Error adding node: {e}")

    def add_relationship(self, source: str, target: str, relation_type: str):
        try:
            query = f"""
            MATCH (a:Entity), (b:Entity)
            WHERE a.id = '{source}' AND b.id = '{target}'
            CREATE (a)-[:RELATED_TO {{ relation_type: '{relation_type}' }}]->(b)
            """
            self.execute_query(query)
        except Exception as e:
            logger.info(f"Error adding relationship: {e}")
            return

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
            query = f"""
            MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
            WHERE a.id = '{source}' AND b.id = '{target}'
            SET r.relation_type = '{relation_type}'
            """

            self.execute_query(query)
            logger.info(
                f"Relationship from '{source}' to '{target}' updated successfully."
            )
        except Exception as e:
            logger.info(f"Error modifying relationship: {e}")
            return

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
                keywords = en.get("search_keywords", [])
                keywords_str = (
                    " ".join(keywords) if isinstance(keywords, list) else keywords
                )

                text = f"Entity: {en.get('id', '')} | Type: {en.get('type', '')} | Keywords: {keywords_str}"
                query_embedding = self._compute_embedding(text)

                query = """
                CALL QUERY_VECTOR_INDEX('Entity', 'Entity_embedding_idx', $query_embedding, 10)
                YIELD node, distance
                WITH node AS n, (1 - distance) AS score
                WHERE score > 0.6
                OPTIONAL MATCH (n)-[r:RELATED_TO]-(m:Entity)
                WITH n, score, collect(r.relation_type + ' with ' + m.id) AS connections
                RETURN n.id AS id, n.type AS type, n.description AS description, 
                    connections AS relations, score AS similarity_score
                ORDER BY similarity_score DESC
                """
                result = self.conn.execute(
                    query, parameters={"query_embedding": query_embedding}
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

        prompt = f"{prompt}\nExisting Knowledge:\n{existing_knowledge}\nNew Knowledge:\n{new_knowledge}\n"

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
