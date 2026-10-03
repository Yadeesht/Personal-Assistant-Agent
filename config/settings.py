import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# API keys and provider endpoints
# -----------------------------------------------------------------------------
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AZURE_AI_CREDENTIAL = os.getenv("AZURE_AI_CREDENTIAL")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

# -----------------------------------------------------------------------------
# Default model and request settings
# -----------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# Default thread for terminal-based sessions
DEFAULT_THREAD_ID = os.getenv("DEFAULT_THREAD_ID", "default_thread")
