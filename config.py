import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Path where Excel files are stored. By default, use project root where your .xlsx files are.
EXCEL_DATA_DIR = os.environ.get("EXCEL_DATA_DIR", str(PROJECT_ROOT))

# Vector store configuration (Supabase Postgres + pgvector)
# NOTE: This backend expects DATABASE_URL to point to a Postgres database (e.g. Supabase).
PGVECTOR_TABLE = os.environ.get("PGVECTOR_TABLE", "rag_documents")
# text-embedding-3-small is 1536 dims. If you change OPENAI_EMBEDDING_MODEL, update this.
PGVECTOR_DIM = int(os.environ.get("PGVECTOR_DIM", "1536"))

# CORS configuration
# Comma-separated list of allowed origins for the frontend (Vercel domain, localhost, etc).
# Example: "https://your-app.vercel.app,http://localhost:3000"
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")

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
