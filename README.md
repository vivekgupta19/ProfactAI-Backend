# ExcelAI RAG Backend (Flask + ChromaDB + OpenAI)

This backend turns your Excel files into a Retrieval-Augmented Generation (RAG) system:

- Upload Excel files from the frontend.
- Each row (across all sheets) is converted to text + metadata.
- Text is embedded with OpenAI embeddings.
- Embeddings are stored in a local **ChromaDB** vector store.
- LLM (ChatGPT) answers questions using the most relevant rows as context.
- Supports both regular and **streaming** chat responses.

---

## 1. Folder structure

From your project root (`/Users/hamzaarfan/Desktop/excelai`):

```text
backend/
  app.py               # Flask app, API routes
  config.py            # Paths, OpenAI models, global config
  requirements.txt     # Python dependencies
  README.md            # This file

  rag/
    __init__.py        # RAG package marker
    ingestion.py       # Excel → text → embeddings → Chroma
    vectorstore.py     # ChromaDB client helpers
    llm.py             # OpenAI (embeddings + chat, streaming)

excelai/               # React frontend (existing)
*.xlsx / *.xls         # Your Excel data files (optional: for bulk reindex)
```

---

## 2. Environment setup

### 2.1 Create a virtual environment & install deps

From the project root (`/Users/hamzaarfan/Desktop/excelai`):

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 2.2 Set environment variables

**Never hardcode your OpenAI key in code.** Use environment variables instead.

On macOS / Linux (bash/zsh):

```bash
# In your terminal, before running the backend
export OPENAI_API_KEY="sk-...your-key-here..."

# Optional overrides (defaults are usually fine):
# Directory where Excel files are read from when calling /api/reindex
export EXCEL_DATA_DIR="/Users/hamzaarfan/Desktop/excelai"

# Directory where ChromaDB stores its local database
export CHROMA_DB_DIR="/Users/hamzaarfan/Desktop/excelai/chroma_db"

# Default number of neighbors for search / RAG
export RAG_TOP_K=5

# Optional model overrides
# export OPENAI_MODEL="gpt-4o-mini"
# export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
```

`config.py` reads these values and will raise an error if `OPENAI_API_KEY` is missing.

---

## 3. Running the backend server

From the project root:

```bash
# 1) Activate venv (if not already)
cd /Users/hamzaarfan/Desktop/excelai
source venv/bin/activate

# 2) Ensure env vars are set (especially OPENAI_API_KEY)
export OPENAI_API_KEY="sk-...your-key-here..."

# 3) Run the Flask app
cd backend
python app.py
```

The server will start on:

```text
http://localhost:5001
```

You can verify it with:

```bash
curl http://localhost:5001/api/health
```

Expected JSON response (example):

```json
{"status": "ok", "indexed_rows": 0}
```

---

## 4. ChromaDB: local vs cloud

This project uses **ChromaDB** as the vector database.

### 4.1 How it’s configured here

In `backend/rag/vectorstore.py`:

- Uses `chromadb.PersistentClient(path=CHROMA_DB_DIR)`.
- `CHROMA_DB_DIR` defaults to `./chroma_db` under the project root.
- Data is stored **locally on disk** (SQLite + Parquet under the hood).

So in this setup:

- **Chroma is local**, not cloud.
- If you stop and restart the server, your vector index is preserved on disk as long as the `chroma_db` folder is kept.
- This is ideal for local development, R&D, and small production setups on a single machine.

### 4.2 If you want cloud / managed vector DB

If you later want cloud-hosted vector DB (multi-machine, autoscaling, etc.), you could swap `vectorstore.py` to use:

- **Pinecone**, **Qdrant Cloud**, **Weaviate Cloud**, etc.
- Or Chroma Cloud (managed offering) instead of the local `PersistentClient`.

That would involve:

- Replacing `chromadb.PersistentClient` with the cloud client.
- Adjusting `add_documents` / `query_documents` accordingly.

For now, this backend is intentionally set up with **local Chroma** for simplicity and zero extra infra cost.

---

## 5. Excel ingestion and indexing

There are **two ways** to get Excel data into Chroma:

1. **Filesystem-based bulk indexing**: `/api/reindex`
2. **Upload-based indexing (recommended for your UI)**: `/api/upload_excel`

### 5.1 Shared logic

`backend/rag/ingestion.py` does the heavy lifting:

- For each Excel file (`.xlsx` / `.xls`):
  - Reads **all sheets** with `pandas.read_excel(..., sheet_name=None, dtype=str)`.
  - For each row in each sheet:
    - Builds a text representation, e.g.

      ```text
      File: RFP_02.xlsx | Sheet: Requirements | ColumnA: valueA | ColumnB: valueB | ...
      ```

    - Skips empty/NaN values.
    - Adds metadata:

      ```json
      {
        "file": "RFP_02.xlsx",
        "sheet": "Requirements",
        "row_index": 10,
        "num_columns": 12
      }
      ```

    - Assigns an ID: `"RFP_02.xlsx::Requirements::10"`.

- Rows are batched (size 100 by default), embedded with OpenAI, and stored in Chroma.

Core helpers:

- `_row_to_text(file_name, sheet_name, row_index, row)` – convert a row to text.
- `_index_sheets(file_name, sheets)` – shared logic to index all sheets of a single Excel file.

### 5.2 Bulk reindex from disk: `/api/reindex`

If you have Excel files already on disk under `EXCEL_DATA_DIR`, you can build the index in one shot.

**Endpoint**: `POST /api/reindex`

- Request body (optional):

  ```json
  { "rebuild": true }
  ```

  - `rebuild=true` (default): drops and recreates the Chroma collection, then reindexes all Excel files.
  - `rebuild=false`: appends to the existing collection.

- Response example:

  ```json
  {
    "status": "success",
    "stats": {
      "excel_files_indexed": ["/path/to/RFP_02.xlsx", "..."],
      "per_file_stats": [
        {"file_name": "RFP_02.xlsx", "total_rows_seen": 300, "documents_added": 300}
      ],
      "total_rows_seen": 300,
      "documents_added": 300,
      "collection_name": "excel_rag_collection"
    }
  }
  ```

### 5.3 Upload-based indexing: `/api/upload_excel` (for your drag & drop)

This is the mode you described: the user drops an Excel file in the frontend, and that file is **immediately indexed into Chroma**.

**Endpoint**: `POST /api/upload_excel`

- Content-Type: `multipart/form-data`
- Form field: `file` – the uploaded Excel (`.xlsx` or `.xls`).

Example `curl`:

```bash
curl -X POST http://localhost:5001/api/upload_excel \
  -F "file=@/path/to/RFP_02.xlsx"
```

Success response example:

```json
{
  "status": "success",
  "stats": {
    "file_name": "RFP_02.xlsx",
    "total_rows_seen": 300,
    "documents_added": 300,
    "collection_name": "excel_rag_collection"
  }
}
```

Notes:

- This **does not reset** the collection; it **appends** rows to the existing index.
- You can upload multiple Excel files over time; all of them become part of the same knowledge base.

---

## 6. Semantic search endpoint

For diagnostics or building search UIs, you have a pure semantic search API.

**Endpoint**: `POST /api/search`

**Request body**:

```json
{
  "query": "What is the implementation timeline?",
  "top_k": 5
}
```

**Response example**:

```json
{
  "query": "What is the implementation timeline?",
  "results": [
    {
      "id": "RFP_02.xlsx::Requirements::10",
      "document": "File: RFP_02.xlsx | Sheet: Requirements | ...",
      "metadata": {
        "file": "RFP_02.xlsx",
        "sheet": "Requirements",
        "row_index": 10,
        "num_columns": 12
      },
      "distance": 0.123
    }
  ]
}
```

---

## 7. Chat endpoints (LLM + RAG)

### 7.1 Non-streaming chat: `/api/chat`

**Endpoint**: `POST /api/chat`

**Request body**:

```json
{
  "question": "Summarize the key requirements and deliverables.",
  "top_k": 5,
  "use_rag": true
}
```

- `use_rag = true` (default): embeds the question, retrieves `top_k` rows from Chroma, and feeds them into ChatGPT as context.
- `use_rag = false`: calls ChatGPT **without** Chroma context (general knowledge only).

**Response example**:

```json
{
  "question": "Summarize the key requirements and deliverables.",
  "answer": "Here is a summary of key requirements...",
  "sources": [
    {
      "id": "RFP_02.xlsx::Requirements::10",
      "text": "File: RFP_02.xlsx | Sheet: Requirements | ColumnA: ...",
      "metadata": {
        "file": "RFP_02.xlsx",
        "sheet": "Requirements",
        "row_index": 10,
        "num_columns": 12
      }
    }
  ]
}
```

This is simple to use with `fetch` on the frontend when you don’t need streaming.

### 7.2 Streaming chat: `/api/chat_stream`

For a more dynamic UI (incremental answer display), use the streaming version.

**Endpoint**: `POST /api/chat_stream`

**Request body** (same shape as `/api/chat`):

```json
{
  "question": "Summarize the key requirements and deliverables.",
  "top_k": 5,
  "use_rag": true
}
```

**Response format**:

- `Content-Type: text/plain`
- A **stream** of newline-delimited JSON objects (JSONL), for example:

  ```text
  {"type": "meta", "data": {"question": "...", "sources": [...]}}
  {"type": "delta", "data": "First part of the answer "}
  {"type": "delta", "data": "next chunk of answer "}
  {"type": "delta", "data": "..."}
  ```

The first line (`type: "meta"`) contains:

- `question`: the original question.
- `sources`: the same context chunks that would be returned by `/api/chat`.

Each subsequent line with `type: "delta"` contains **partial answer text**. Concatenate them on the frontend to build the full answer.

#### 7.2.1 Example frontend consumption (React / browser `fetch`)

```ts
async function streamChat(question: string, onDelta: (chunk: string) => void, onMeta?: (meta: any) => void) {
  const response = await fetch("http://localhost:5001/api/chat_stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: 5, use_rag: true }),
  });

  if (!response.body) throw new Error("No response body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    let newlineIndex;
    while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (!line) continue;

      const msg = JSON.parse(line);
      if (msg.type === "meta" && onMeta) {
        onMeta(msg.data);
      } else if (msg.type === "delta") {
        onDelta(msg.data);
      }
    }
  }
}
```

In your React dashboard component, you can:

- Maintain `answer` state and append each `delta`:

  ```ts
  const [answer, setAnswer] = useState("");

  async function handleAsk(question: string) {
    setAnswer("");
    await streamChat(
      question,
      (chunk) => setAnswer((prev) => prev + chunk),
      (meta) => {
        // meta.sources contains the retrieved rows from Chroma
        console.log("sources", meta.sources);
      }
    );
  }
  ```

This will render the answer in a **streamed** way as the model responds.

---

## 8. Frontend integration overview

### 8.1 Upload Excel from your drag & drop UI

In your existing React landing/dashboard code where you handle file drop:

1. When the user drops an Excel file, send it to `/api/upload_excel`:

   ```ts
   async function uploadExcel(file: File) {
     const formData = new FormData();
     formData.append("file", file);

     const res = await fetch("http://localhost:5001/api/upload_excel", {
       method: "POST",
       body: formData,
     });

     const json = await res.json();
     console.log("upload stats", json);
   }
   ```

2. After a successful upload, show a success message like "File indexed into knowledge base".

### 8.2 Ask questions on the dashboard

On your dashboard page:

- Add a text area / input for the question.
- A "Ask" button that calls either:
  - `/api/chat` (simple, non-streaming) or
  - `/api/chat_stream` (recommended for dynamic streaming UI).

With `/api/chat` (simpler):

```ts
async function askQuestion(question: string) {
  const res = await fetch("http://localhost:5001/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: 5, use_rag: true }),
  });
  const json = await res.json();
  setAnswer(json.answer);
  setSources(json.sources);
}
```

With `/api/chat_stream`, use the `streamChat` helper above for a more "live" chat feel.

---

## 9. Endpoints summary

- `GET  /api/health`
  - Health check, returns `{ status, indexed_rows }`.

- `POST /api/reindex`
  - Bulk reindex of all Excel files under `EXCEL_DATA_DIR`.
  - Body: `{ "rebuild": true | false }`.

- `POST /api/upload_excel`
  - Upload and index a single Excel file via multipart/form-data.
  - Field: `file` (must be `.xlsx` or `.xls`).

- `POST /api/search`
  - Semantic search over the Chroma index.
  - Body: `{ "query": string, "top_k": number }`.

- `POST /api/chat`
  - Non-streaming RAG chat.
  - Body: `{ "question": string, "top_k"?: number, "use_rag"?: boolean }`.

- `POST /api/chat_stream`
  - Streaming RAG chat (JSONL).
  - Body: `{ "question": string, "top_k"?: number, "use_rag"?: boolean }`.

---

## 10. What’s happening under the hood (high-level R&D view)

1. **Ingestion**
   - Excel files (multi-sheet, heterogeneous columns) are normalized row-wise into rich text strings.
   - Each row gets metadata (file, sheet, row index, column count).
   - Batches of rows are embedded using **OpenAI embeddings**.
   - Embeddings + text + metadata are stored in **ChromaDB**.

2. **Retrieval**
   - For a user query, we compute its embedding.
   - Chroma is queried for the `top_k` nearest neighbors.
   - Retrieved rows (text + metadata) become the **context** for the LLM.

3. **Generation (RAG)**
   - System prompt tells ChatGPT to answer **only** from the provided context and not to hallucinate.
   - Context is formatted as labeled sources (with file/sheet/row) for traceability.
   - The model generates an answer:
     - Either as a single completion (`/api/chat`).
     - Or as a stream of chunks (`/api/chat_stream`).

4. **Frontend UX**
   - User drops Excel → `/api/upload_excel` → rows inserted into Chroma.
   - User asks question on dashboard → `/api/chat_stream` → answer text appears incrementally.
   - Optional: show source rows (file/sheet/row) beneath the answer for transparency.

This gives you an end-to-end RAG system: Excel → vector DB (Chroma) → semantic search + LLM → streamed answers, all powered by your ChatGPT API key.
