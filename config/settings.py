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

# Per-model request options, set in .env only for models that need them.
# MODEL_TEMPERATURE: some models accept only their default (gpt-6-luna: 1). Unset = the
#   client default (0.7).
# MODEL_TOOL_REASONING_EFFORT: reasoning effort for calls that offer tools. gpt-6-luna
#   rejects tools on Chat Completions unless this is "none". Leave unset for models
#   without reasoning (gpt-4.1-mini rejects the parameter). Calls without tools
#   (summarizer, knowledge-graph extraction) keep the model's default reasoning.
MODEL_TEMPERATURE = (
    float(os.getenv("MODEL_TEMPERATURE")) if os.getenv("MODEL_TEMPERATURE") else None
)
MODEL_TOOL_REASONING_EFFORT = os.getenv("MODEL_TOOL_REASONING_EFFORT") or None

# Tools that act outside the app and cannot be undone. Before one runs, the app pauses
# and shows the user exactly what it would do; it runs only if they type yes
# (core/confirm.py). Calendar invites (create_event with attendees) could be added here.
CONFIRM_BEFORE_TOOLS = {"send_email"}

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

# -----------------------------------------------------------------------------
# Long-term memory
# -----------------------------------------------------------------------------
# Look up related knowledge-graph facts and past conversations for every user message
# and show them to the agents. Set MEMORY_RECALL=false to turn it off (e.g. for evals).
MEMORY_RECALL = os.getenv("MEMORY_RECALL", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
