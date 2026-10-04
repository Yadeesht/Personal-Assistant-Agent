import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"
MEMORY_DB = DATA_DIR / "memory.db"
VECTOR_DB = DATA_DIR / "embeddings"
KNOWLEDGE_GRAPH_DB = DATA_DIR / "knowledge_graph_db" / "knowledge_graph.db"
EPISODIC_RAG_DB = DATA_DIR / "episodic_rag_db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_GRAPH_DB.parent.mkdir(parents=True, exist_ok=True)
EPISODIC_RAG_DB.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# API keys and provider endpoints
# -----------------------------------------------------------------------------
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AZURE_AI_CREDENTIAL = os.getenv("AZURE_AI_CREDENTIAL")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -----------------------------------------------------------------------------
# Default model and request settings
# -----------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# -----------------------------------------------------------------------------
# Embedding model paths
# -----------------------------------------------------------------------------
EMBEDDING_BGE_MODEL_PATH = BASE_DIR / "models" / "bge-small"
EMBEDDING_GTE_MODEL_PATH = BASE_DIR / "models" / "gte-base"

# -----------------------------------------------------------------------------
# Token and conversation defaults
# -----------------------------------------------------------------------------
MAX_TOKENS = 2000
TOKEN_STRATEGY = "last"

# Default thread for terminal-based sessions
DEFAULT_THREAD_ID = os.getenv("DEFAULT_THREAD_ID", "default_thread")
