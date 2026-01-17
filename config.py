import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Path where Excel files are stored. By default, use project root where your .xlsx files are.
EXCEL_DATA_DIR = os.environ.get("EXCEL_DATA_DIR", str(PROJECT_ROOT))

# ChromaDB Configuration
# For cloud ChromaDB, these will be used only if CHROMA_API_KEY is not set
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", str(PROJECT_ROOT / "chroma_db"))

# ChromaDB Cloud Configuration (Official Chroma Cloud)
CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY")
CHROMA_TENANT = os.environ.get("CHROMA_TENANT")
CHROMA_DATABASE = os.environ.get("CHROMA_DATABASE")

# OpenAI configuration (ChatGPT + embeddings)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.environ.get(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

# RAG defaults
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))


def validate_config():
    """Validate required configuration"""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set it in your environment before running the backend."
        )
    
    # Validate ChromaDB configuration
    if CHROMA_API_KEY and CHROMA_TENANT and CHROMA_DATABASE:
        print(f"✅ ChromaDB Cloud Mode")
        print(f"   Tenant: {CHROMA_TENANT}")
        print(f"   Database: {CHROMA_DATABASE}")
    else:
        print(f"⚠️  ChromaDB Local Mode: {CHROMA_DB_DIR}")
        print("   Set CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE to use Chroma Cloud")


def get_chroma_mode() -> str:
    """Return the current ChromaDB mode"""
    if CHROMA_API_KEY and CHROMA_TENANT and CHROMA_DATABASE:
        return "cloud"
    return "local"