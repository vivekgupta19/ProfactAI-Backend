import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Path where Excel files are stored. By default, use project root where your .xlsx files are.
EXCEL_DATA_DIR = os.environ.get("EXCEL_DATA_DIR", str(PROJECT_ROOT))

# ChromaDB persistent directory
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", str(PROJECT_ROOT / "chroma_db"))

# OpenAI configuration (ChatGPT + embeddings)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.environ.get(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

# RAG defaults
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))


def validate_config():
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set it in your environment before running the backend."
        )
