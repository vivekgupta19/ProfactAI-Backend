# ExcelAI Backend (Flask + Supabase Postgres pgvector + OpenAI)

This backend turns your Excel rows into a Retrieval-Augmented Generation (RAG) system:

- Upload Excel files from the frontend.
- Each row is converted to text + metadata.
- Text is embedded with OpenAI embeddings.
- Embeddings are stored in **Supabase Postgres using pgvector**.
- Chat answers use the most relevant rows as context (streaming supported).

## Local development

From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### Required environment variables

```bash
export OPENAI_API_KEY="sk-..."
export DATABASE_URL="postgresql://... (Supabase connection string, usually with sslmode=require)"
export JWT_SECRET_KEY="change-me"
```

### Recommended environment variables

```bash
# Frontend origins allowed to call the API (Vercel + localhost)
export CORS_ORIGINS="http://localhost:3000,https://your-app.vercel.app"

# Where /api/reindex reads Excel files from (optional)
export EXCEL_DATA_DIR="/Users/hamzaarfan/Desktop/excelai"

# RAG defaults
export RAG_TOP_K=5

# pgvector settings (update if you change OPENAI_EMBEDDING_MODEL)
export PGVECTOR_TABLE="rag_documents"
export PGVECTOR_DIM=1536
```

### Run

```bash
cd backend
export PORT=5001
python app.py
```

## Production (Railway)

Recommended start command (from repo root):

```bash
cd backend && gunicorn -w 2 -b 0.0.0.0:$PORT "app:create_app()"
```

## Reindex / ingestion

- `POST /api/upload_document`: upload + index a single file into pgvector
- `POST /api/reindex`: rebuild or append an index from `EXCEL_DATA_DIR`
- `POST /api/chat` and `POST /api/chat_stream`: ask questions with RAG
