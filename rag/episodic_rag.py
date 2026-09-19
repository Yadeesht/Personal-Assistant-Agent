import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import uuid
import re
import aiosqlite
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np


MIN_MESSAGE_TOKENS = 25
MIN_CHUNK_TOKENS = 200
MAX_CHUNK_TOKENS = (
    1000  # this threshold need to optimize based on performance now done heeeee
)
MAX_TIME_GAP_SECONDS = 3600

from config.settings import EMBEDDING_GTE_MODEL_PATH, MEMORY_DB, EPISODIC_RAG_DB
from utils.helper import setup_logger, count_tokens

logger = setup_logger(__name__)


class EpisodicRAG:
    def __init__(self, db_path=MEMORY_DB):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._model = None

    PREFIX_PATTERN = re.compile(
        r"^(TALK TO USER:|FINAL ANSWER:|CLARIFICATION NEEDED:)\s*", re.IGNORECASE
    )
    DECORATOR_PATTERN = re.compile(r"[-=_*|]{3,}")
    LOG_HEADER_PATTERN = re.compile(r"^\[.*?\] \[.*?\]\s*")

    def clean_messages_for_chunk(self, text: str):
        if not text:
            return None

        if "<|channel|>" in text:
            return None
        if text.startswith("Routing to:"):
            return None

        text = self.LOG_HEADER_PATTERN.sub("", text)
        text = self.PREFIX_PATTERN.sub("", text)
        text = self.DECORATOR_PATTERN.sub(" ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        cleaned_text = text.strip()
        if not cleaned_text:
            return None

        return cleaned_text

    def _split_text_to_chunks(self, text_lines, timestamp, actors):
        task_uuid = str(uuid.uuid4())
        generated_chunks = []

        cleaned_lines = []
        for line in text_lines:
            clean = self.clean_messages_for_chunk(line)
            if clean:
                cleaned_lines.append(clean)

        if not cleaned_lines:
            return []

        full_text = "\n".join(cleaned_lines)
        if count_tokens(full_text) <= MAX_CHUNK_TOKENS:
            return [
                {
                    "id": str(uuid.uuid4()),
                    "content": full_text,
                    "metadata": {
                        "timestamp": timestamp,
                        "task_id": task_uuid,
                        "part": 1,
                        "total_parts": 1,
                        "actors": actors,
                    },
                }
            ]

        current_lines = []
        current_tokens = 0
        part = 1

        for line in cleaned_lines:
            line_tokens = count_tokens(line)

            if line_tokens > MAX_CHUNK_TOKENS:
                if current_lines:
                    generated_chunks.append(
                        {
                            "id": str(uuid.uuid4()),
                            "content": "\n".join(current_lines),
                            "metadata": {
                                "timestamp": timestamp,
                                "task_id": task_uuid,
                                "part": part,
                                "total_parts": 0,
                                "actors": actors,
                            },
                        }
                    )
                    part += 1
                    current_lines = []
                    current_tokens = 0

                safe_char_limit = MAX_CHUNK_TOKENS * 4
                truncated_content = (
                    line[:safe_char_limit] + "\n...(truncated due to size)..."
                )

                generated_chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "content": truncated_content,
                        "metadata": {
                            "timestamp": timestamp,
                            "task_id": task_uuid,
                            "part": part,
                            "actors": actors,
                            "oversized": True,
                        },
                    }
                )
                part += 1
                continue

            if current_tokens + line_tokens > MAX_CHUNK_TOKENS:
                generated_chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "content": "\n".join(current_lines),
                        "metadata": {
                            "timestamp": timestamp,
                            "task_id": task_uuid,
                            "part": part,
                            "total_parts": 0,
                            "actors": actors,
                        },
                    }
                )
                current_lines = [line]
                current_tokens = line_tokens
                part += 1

            else:
                current_lines.append(line)
                current_tokens += line_tokens

        if current_lines:
            generated_chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "content": "\n".join(current_lines),
                    "metadata": {
                        "timestamp": timestamp,
                        "task_id": task_uuid,
                        "part": part,
                        "total_parts": 0,
                        "actors": actors,
                    },
                }
            )

        final_total_parts = len(generated_chunks)
        for chunk in generated_chunks:
            chunk["metadata"]["total_parts"] = final_total_parts

        return generated_chunks

    @property
    def model(self):
        if self._model is None:
            try:
                if not Path(EMBEDDING_GTE_MODEL_PATH).exists():
                    self._model = SentenceTransformer("unsloth/gte-modernbert-base")
                    self._model.save(EMBEDDING_GTE_MODEL_PATH)
                self._model = SentenceTransformer(EMBEDDING_GTE_MODEL_PATH)

                logger.info("gte-modernbert-base model loaded successfully.")
            except Exception as e:
                logger.info(f"Error loading gte-modernbert-base model: {e}")
                raise e
        return self._model

    def _embedding_chunk(self, chunk):
        try:
            content = chunk["content"] if isinstance(chunk, dict) else chunk
            embedding = self.model.encode(content, show_progress_bar=False)
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    async def custom_text_splitters(self, past_summary_date=None):
        try:
            if not past_summary_date:
                logger.warning(
                    "Past summary date is not set. No data will be retrieved."
                )
                return []
            else:
                past_summary_date = datetime.fromtimestamp(
                    past_summary_date
                ).isoformat()
                logger.info(f"Last summary date set to: {past_summary_date}")

            query = """
                SELECT timestamp, actor, message, metadata
                FROM human_logs 
                WHERE timestamp > ? 
                AND actor NOT IN ('supervisor_routing', 'summerizer_node')
                AND COALESCE(json_extract(metadata, '$.type'), '') != 'tool_call'
                ORDER BY timestamp ASC
            """

            async with aiosqlite.connect(self.path) as db:
                async with db.execute(query, (past_summary_date,)) as cursor:
                    rows = await cursor.fetchall()
                    logger.info(f"Processing {len(rows)} raw logs...")

            if len(rows) == 0:
                logger.info("No new logs to process since the last summary.")
                return []

            episodes = []
            current_lines = []
            current_start_ts = None
            current_ts_obj = None
            actors = set()

            for timestamp, actor, message, _metadata in rows:
                try:
                    ts_obj = datetime.fromisoformat(timestamp)
                except ValueError:
                    continue

                if actor == "Human_node":
                    if current_lines:
                        episodes.append(
                            {
                                "ts_obj": current_ts_obj,
                                "timestamp": current_start_ts,
                                "lines": current_lines,
                                "actors": list(actors),
                            }
                        )
                    current_lines = [f"User: {message}"]
                    current_start_ts = timestamp
                    current_ts_obj = ts_obj
                    actors = set()

                elif actor == "supervisor_task_response":
                    current_lines.append(f"Assistant: {message}")
                    actors.add("supervisor")
                    if current_lines:
                        episodes.append(
                            {
                                "ts_obj": current_ts_obj,
                                "timestamp": current_start_ts,
                                "lines": current_lines,
                                "actors": list(actors),
                            }
                        )
                        current_lines = []
                        current_start_ts = None
                        actors = set()
                else:
                    if current_lines:
                        current_lines.append(f"{actor}: {message}")
                        actors.add(actor)

            if current_lines:
                episodes.append(
                    {
                        "ts_obj": current_ts_obj,
                        "timestamp": current_start_ts,
                        "lines": current_lines,
                        "actors": list(actors),
                    }
                )

            final_chunks = []

            for episode in episodes:
                ep_text = "\n".join(episode["lines"])
                ep_clean = self.clean_messages_for_chunk(ep_text)

                current_actors = episode.get("actors")
                if not current_actors:
                    current_actors = ["supervisor"]

                if not ep_clean:
                    continue

                ep_tokens = count_tokens(ep_clean)
                has_tool = any(
                    "__Tool Action__" in line or "Using tools" in line
                    for line in episode["lines"]
                )

                if ep_tokens < MIN_MESSAGE_TOKENS and not has_tool:
                    continue

                elif (
                    ep_tokens < MIN_CHUNK_TOKENS
                ):  # Why backward? Because a small follow-up usually relates to the previous big thought.
                    merged = False
                    last_chunk = None
                    if final_chunks:
                        last_chunk = final_chunks[-1]

                    try:
                        if last_chunk:
                            last_ts = datetime.fromisoformat(
                                last_chunk["metadata"]["timestamp"]
                            )
                            time_gap = (episode["ts_obj"] - last_ts).total_seconds()
                        else:
                            time_gap = 999999
                    except (ValueError, TypeError):
                        time_gap = 999999

                    if time_gap < MAX_TIME_GAP_SECONDS and last_chunk:
                        current_last_tokens = count_tokens(last_chunk["content"])

                        if current_last_tokens + ep_tokens <= MAX_CHUNK_TOKENS:
                            last_chunk["content"] += "\n" + ep_clean

                            existing_actors = set(last_chunk["metadata"]["actors"])
                            existing_actors.update(current_actors)
                            last_chunk["metadata"]["actors"] = list(existing_actors)

                            merged = True

                    if not merged:
                        task_uuid = str(uuid.uuid4())
                        final_chunks.append(
                            {
                                "id": str(uuid.uuid4()),
                                "content": ep_clean,
                                "embedding": None,
                                "metadata": {
                                    "timestamp": episode["timestamp"],
                                    "task_id": task_uuid,
                                    "part": 1,
                                    "total_parts": 1,
                                    "actors": current_actors,
                                },
                            }
                        )

                else:
                    if ep_tokens > MAX_CHUNK_TOKENS:
                        chunks = self._split_text_to_chunks(
                            episode["lines"], episode["timestamp"], current_actors
                        )
                        final_chunks.extend(chunks)
                    else:
                        task_uuid = str(uuid.uuid4())
                        final_chunks.append(
                            {
                                "id": str(uuid.uuid4()),
                                "content": ep_clean,
                                "embedding": None,
                                "metadata": {
                                    "timestamp": episode["timestamp"],
                                    "task_id": task_uuid,
                                    "part": 1,
                                    "total_parts": 1,
                                    "actors": current_actors,
                                },
                            }
                        )

            for i in range(len(final_chunks)):
                final_chunks[i]["metadata"]["prev_id"] = None
                final_chunks[i]["metadata"]["next_id"] = None
                final_chunks[i]["embedding"] = self._embedding_chunk(
                    final_chunks[i]["content"]
                )

                try:
                    curr_ts = datetime.fromisoformat(
                        final_chunks[i]["metadata"]["timestamp"]
                    )
                except (ValueError, TypeError):
                    continue

                if i > 0:
                    try:
                        prev_ts = datetime.fromisoformat(
                            final_chunks[i - 1]["metadata"]["timestamp"]
                        )
                        if (curr_ts - prev_ts).total_seconds() < MAX_TIME_GAP_SECONDS:
                            final_chunks[i]["metadata"]["prev_id"] = final_chunks[
                                i - 1
                            ]["id"]
                    except (ValueError, TypeError):
                        pass

                if i < len(final_chunks) - 1:
                    try:
                        next_ts = datetime.fromisoformat(
                            final_chunks[i + 1]["metadata"]["timestamp"]
                        )
                        if (next_ts - curr_ts).total_seconds() < MAX_TIME_GAP_SECONDS:
                            final_chunks[i]["metadata"]["next_id"] = final_chunks[
                                i + 1
                            ]["id"]
                    except (ValueError, TypeError):
                        pass

            logger.info(
                f"Chunking Complete. Generated {len(final_chunks)} linked chunks."
            )
            return final_chunks
        except Exception as e:
            logger.error(f"Error during custom text splitting: {e}")
            return []

    def index_creation(self, final_chunks):
        client = None
        try:
            client = QdrantClient(path=EPISODIC_RAG_DB)

            collection_name = "episodic_chunks"

            if not client.collection_exists(collection_name):
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=768, distance=models.Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfig(
                        m=16, ef_construct=100, full_scan_threshold=10000
                    ),
                )

            points = []
            for chunk in final_chunks:
                payload = {
                    "content": chunk["content"],
                    "timestamp": chunk["metadata"]["timestamp"],
                    "task_id": chunk["metadata"]["task_id"],
                    "part": chunk["metadata"]["part"],
                    "total_parts": chunk["metadata"].get("total_parts", 0),
                    "actors": chunk["metadata"]["actors"],
                    "prev_id": chunk["metadata"]["prev_id"],
                    "next_id": chunk["metadata"]["next_id"],
                }

                points.append(
                    models.PointStruct(
                        id=chunk["id"],
                        vector=chunk["embedding"].tolist(),
                        payload=payload,
                    )
                )

            client.upsert(collection_name=collection_name, points=points)

        except Exception as e:
            logger.error(f"Error during index creation: {e}")
        finally:
            if client is not None:
                client.close()

    def retrieve_chunks(self, query, conditions=None, top_k=5):
        client = None
        try:
            if conditions is None:
                conditions = {}

            query_vector = self._embedding_chunk(query)
            client = QdrantClient(path=EPISODIC_RAG_DB)

            must = []
            if conditions.get("actors"):
                actor_list = conditions["actors"]
                if isinstance(actor_list, str):
                    actor_list = [actor_list]

                must.append(
                    models.FieldCondition(
                        key="actors", match=models.MatchAny(any=actor_list)
                    )
                )

            if conditions.get("start_time") and conditions.get("end_time"):
                # "timestamp" is stored as an ISO-8601 string, so a numeric
                # Range would raise a validation error — DatetimeRange accepts
                # ISO-8601 strings directly.
                must.append(
                    models.FieldCondition(
                        key="timestamp",
                        range=models.DatetimeRange(
                            gte=conditions["start_time"], lte=conditions["end_time"]
                        ),
                    )
                )

            query_filter = models.Filter(must=must) if must else None

            search_result = client.query_points(
                collection_name="episodic_chunks",
                query=query_vector.tolist(),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            ).points

            expanded_results = []
            seen_ids = set()

            for hit in search_result:
                if hit.id in seen_ids:
                    continue

                chunk = hit.payload
                score = hit.score

                if chunk.get("total_parts", 0) > 1:
                    siblings = client.scroll(
                        collection_name="episodic_chunks",
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="task_id",
                                    match=models.MatchValue(value=chunk["task_id"]),
                                )
                            ]
                        ),
                        limit=100,
                    )[0]

                    siblings.sort(key=lambda x: x.payload["part"])
                    full_context = "\n".join(s.payload["content"] for s in siblings)

                    expanded_results.append(
                        {
                            "context": full_context,
                            "score": score,
                            "type": "reconstructed_task",
                        }
                    )

                    for s in siblings:
                        seen_ids.add(s.id)

                    continue

                if score > 0.75:
                    context_block = [chunk["content"]]

                    prev_id = chunk.get("prev_id")
                    if prev_id:
                        prev_chunk = client.retrieve(
                            collection_name="episodic_chunks",
                            ids=[prev_id],
                        )
                        if prev_chunk:
                            context_block.insert(0, prev_chunk[0].payload["content"])
                            seen_ids.add(prev_chunk[0].id)

                    next_id = chunk.get("next_id")
                    if next_id:
                        next_chunk = client.retrieve(
                            collection_name="episodic_chunks",
                            ids=[next_id],
                        )
                        if next_chunk:
                            context_block.append(next_chunk[0].payload["content"])
                            seen_ids.add(next_chunk[0].id)

                    expanded_results.append(
                        {
                            "context": "\n".join(context_block),
                            "score": score,
                            "type": "expanded_chunk",
                        }
                    )
                    seen_ids.add(hit.id)

                else:
                    expanded_results.append(
                        {
                            "context": chunk["content"],
                            "score": score,
                            "type": "raw_chunk",
                        }
                    )
                    seen_ids.add(hit.id)

            return expanded_results

        except Exception as e:
            logger.info(f"Error during chunk retrieval: {e}")
            return None
        finally:
            if client is not None:
                client.close()
