from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from auth import hash_password, login_user, logout_user, register_user
from auth_fastapi import get_current_user_from_auth_header
from config import CORS_ORIGINS, TOP_K
from database import db
from rag import ingestion, llm, vectorstore


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PROCESSED_EXCEL_DIR = Path(
    os.environ.get("PROCESSED_EXCEL_OUTPUT_DIR", PROJECT_ROOT / "processed_excels")
)
PROCESSED_EXCEL_DIR.mkdir(parents=True, exist_ok=True)

JOB_METADATA: Dict[str, Dict[str, Any]] = {}


def _parse_cors_origins(raw: Optional[str]) -> list[str]:
    origins = [o.strip() for o in (raw or "").split(",") if o.strip()]
    return origins or ["http://localhost:3000", "http://127.0.0.1:3000"]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
    max_age=3600,
)


@app.get("/api/ping")
def ping():
    return {"status": "ok"}


@app.on_event("startup")
def _bootstrap_superadmin_if_enabled() -> None:
    """
    Legacy implementation only created this user in a "run directly" mode.
    Here we keep it opt-in via env so local dev can match that behavior.
    """
    enabled = os.environ.get("BOOTSTRAP_SUPERADMIN_ENABLED", "false").lower() == "true"
    if not enabled:
        return

    email = os.environ.get("BOOTSTRAP_SUPERADMIN_EMAIL", "vivekgupta@profact.ai")
    password = os.environ.get("BOOTSTRAP_SUPERADMIN_PASSWORD", "vivek123")
    org_slug = os.environ.get("BOOTSTRAP_SUPERADMIN_ORG_SLUG", "profact-admin")
    full_name = os.environ.get("BOOTSTRAP_SUPERADMIN_FULL_NAME", "Vivek Gupta")
    role = os.environ.get("BOOTSTRAP_SUPERADMIN_ROLE", "superadmin")

    try:
        existing = db.get_user_by_email(email)
        if existing:
            return

        org = db.get_organization_by_slug(org_slug)
        if not org:
            return

        password_hash = hash_password(password)
        db.create_user(
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            organization_id=org["organizationid"],
            role=role,
        )
        print("✅ Superadmin user created successfully")
    except Exception as e:
        print(f"⚠️  Could not create superadmin: {e}")


# ==================== AUTH ROUTES ====================


@app.post("/api/auth/register")
def register(payload: Dict[str, Any] = Body(...)):
    email = payload.get("email")
    password = payload.get("password")
    full_name = payload.get("fullName")
    organization_name = payload.get("organizationName")

    user_data, error = register_user(email, password, full_name, organization_name)
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Auto-login after registration
    auth_data, login_error = login_user(email, password)
    if login_error:
        raise HTTPException(status_code=500, detail=login_error)

    return {"status": "success", "message": "Registration successful", "data": auth_data}


@app.post("/api/auth/login")
def login(payload: Dict[str, Any] = Body(...)):
    email = payload.get("email")
    password = payload.get("password")

    auth_data, error = login_user(email, password)
    if error:
        raise HTTPException(status_code=401, detail=error)

    return {"status": "success", "data": auth_data}


@app.post("/api/auth/logout")
def logout(user: Dict[str, Any] = Depends(get_current_user_from_auth_header)):
    token = user.get("_token")
    if token:
        logout_user(token)
    return {"status": "success", "message": "Logged out successfully"}


@app.get("/api/auth/me")
def me(user: Dict[str, Any] = Depends(get_current_user_from_auth_header)):
    user_out = dict(user)
    user_out.pop("_token", None)
    return {"status": "success", "data": {"user": user_out}}


# ==================== PROTECTED ROUTES ====================


@app.get("/api/health")
def health(user: Dict[str, Any] = Depends(get_current_user_from_auth_header)):
    try:
        collection = vectorstore.get_user_collection(user["organizationId"])
        count = collection.count()
    except Exception:
        count = 0
    return {
        "status": "ok",
        "indexedRows": count,
        "organizationId": user["organizationId"],
        "organizationName": user["organizationName"],
    }


@app.post("/api/reindex")
def reindex(
    payload: Dict[str, Any] = Body(default={}),
    user: Dict[str, Any] = Depends(get_current_user_from_auth_header),
):
    rebuild = bool(payload.get("rebuild", True))
    try:
        stats = ingestion.build_index(
            user["organizationId"],
            user["userId"],
            rebuild=rebuild,
        )
        return {"status": "success", "stats": stats}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/upload_document")
def upload_document(
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user_from_auth_header),
):
    allowed_extensions = {".xlsx", ".xls", ".csv"}
    file_ext = Path(file.filename or "").suffix.lower()
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    try:
        stats = ingestion.index_uploaded_file(
            file.file,
            file.filename,
            user["organizationId"],
            user["userId"],
        )
        return {"status": "success", "stats": stats}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/search")
def search(
    payload: Dict[str, Any] = Body(...),
    user: Dict[str, Any] = Depends(get_current_user_from_auth_header),
):
    query = payload.get("query")
    top_k = int(payload.get("top_k") or TOP_K)
    if not query:
        raise HTTPException(status_code=400, detail="'query' is required")

    try:
        query_embedding = llm.embed_texts([query])[0]
        result = vectorstore.query_user_documents(
            user["organizationId"],
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        hits = []
        for i in range(len(result["ids"][0])):
            hits.append(
                {
                    "id": result["ids"][0][i],
                    "document": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "distance": result.get("distances", [[None]])[0][i]
                    if result.get("distances")
                    else None,
                }
            )

        return {"query": query, "results": hits}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/chat")
def chat(
    payload: Dict[str, Any] = Body(...),
    user: Dict[str, Any] = Depends(get_current_user_from_auth_header),
):
    question = payload.get("question")
    top_k = int(payload.get("top_k") or TOP_K)
    use_rag = bool(payload.get("use_rag", True))
    if not question:
        raise HTTPException(status_code=400, detail="'question' is required")

    try:
        context_chunks = []
        if use_rag:
            query_embedding = llm.embed_texts([question])[0]
            result = vectorstore.query_user_documents(
                user["organizationId"],
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            for i in range(len(result["ids"][0])):
                context_chunks.append(
                    {
                        "id": result["ids"][0][i],
                        "text": result["documents"][0][i],
                        "metadata": result["metadatas"][0][i],
                    }
                )

        answer = llm.chat_with_context(question, context_chunks)
        return {"question": question, "answer": answer, "sources": context_chunks}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/chat_stream")
def chat_stream(
    payload: Dict[str, Any] = Body(...),
    user: Dict[str, Any] = Depends(get_current_user_from_auth_header),
):
    question = payload.get("question")
    top_k = int(payload.get("top_k") or TOP_K)
    use_rag = bool(payload.get("use_rag", True))
    if not question:
        raise HTTPException(status_code=400, detail="'question' is required")

    try:
        context_chunks = []
        if use_rag:
            query_embedding = llm.embed_texts([question])[0]
            result = vectorstore.query_user_documents(
                user["organizationId"],
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            for i in range(len(result["ids"][0])):
                context_chunks.append(
                    {
                        "id": result["ids"][0][i],
                        "text": result["documents"][0][i],
                        "metadata": result["metadatas"][0][i],
                    }
                )

        def generate():
            header = {"question": question, "sources": context_chunks}
            yield json.dumps({"type": "meta", "data": header}) + "\n"
            for chunk in llm.chat_with_context_stream(question, context_chunks):
                yield json.dumps({"type": "delta", "data": chunk}) + "\n"

        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/document_qa_stream")
def document_qa_stream(
    file: UploadFile = File(...),
    questionColumn: Optional[str] = Form(default=None),
    answerColumn: Optional[str] = Form(default="aiAnswer"),
    batchSize: Optional[str] = Form(default=None),
    maxRows: Optional[str] = Form(default=None),
    topK: Optional[str] = Form(default=None),
    user: Dict[str, Any] = Depends(get_current_user_from_auth_header),
):
    allowed_extensions = {".xlsx", ".xls", ".csv"}
    file_ext = Path(file.filename or "").suffix.lower()
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    try:
        batch_size = int(batchSize) if batchSize else 10
        if batch_size <= 0:
            batch_size = 10
    except ValueError:
        batch_size = 10

    try:
        top_k = int(topK) if topK else TOP_K
        if top_k <= 0:
            top_k = TOP_K
    except ValueError:
        top_k = TOP_K

    max_rows = None
    if maxRows:
        try:
            mr = int(maxRows)
            max_rows = mr if mr > 0 else None
        except ValueError:
            max_rows = None

    try:
        if file_ext == ".csv":
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file has no rows")

    if questionColumn and questionColumn in df.columns:
        q_col = questionColumn
    else:
        q_col = df.columns[0]
        questionColumn = q_col

    job_id = str(uuid.uuid4())
    output_path = PROCESSED_EXCEL_DIR / f"{job_id}{file_ext}"
    JOB_METADATA[job_id] = {
        "originalFilename": file.filename or f"uploaded{file_ext}",
        "outputPath": str(output_path),
        "questionColumn": questionColumn,
        "answerColumn": answerColumn,
        "userId": user["userId"],
        "organizationId": user["organizationId"],
    }

    def generate():
        total_rows = len(df)
        limit = max_rows if max_rows is not None else total_rows
        if limit > total_rows:
            limit = total_rows

        header = {
            "jobId": job_id,
            "totalRows": int(total_rows),
            "limitRows": int(limit),
            "questionColumn": questionColumn,
            "answerColumn": answerColumn,
            "batchSize": int(batch_size),
            "topK": int(top_k),
        }
        yield json.dumps({"type": "meta", "data": header}) + "\n"

        processed_rows = 0

        for start in range(0, limit, batch_size):
            end = min(start + batch_size, limit)
            batch_indices = list(range(start, end))
            if not batch_indices:
                continue

            yield json.dumps(
                {
                    "type": "status",
                    "data": {"event": "batchStart", "fromRow": int(start), "toRow": int(end - 1)},
                }
            ) + "\n"

            for idx in batch_indices:
                try:
                    question_value = df.iloc[idx][q_col]
                except Exception:
                    continue

                if question_value is None:
                    continue

                question_text = str(question_value).strip()
                if not question_text or question_text.lower() in {"nan", "none"}:
                    continue

                context_chunks = []
                try:
                    query_embedding = llm.embed_texts([question_text])[0]
                    result = vectorstore.query_user_documents(
                        user["organizationId"],
                        query_embeddings=[query_embedding],
                        n_results=top_k,
                    )
                    for i in range(len(result["ids"][0])):
                        context_chunks.append(
                            {
                                "id": result["ids"][0][i],
                                "text": result["documents"][0][i],
                                "metadata": result["metadatas"][0][i],
                            }
                        )
                except Exception as e:
                    yield json.dumps(
                        {
                            "type": "warning",
                            "data": {"rowIndex": int(idx), "message": f"RAG retrieval failed: {e}"},
                        }
                    ) + "\n"

                try:
                    answer_text = llm.chat_with_context(question_text, context_chunks)
                except Exception as e:
                    answer_text = f"Error generating answer: {e}"

                df.loc[df.index[idx], answerColumn] = answer_text
                processed_rows += 1

                yield json.dumps(
                    {
                        "type": "rowAnswer",
                        "data": {
                            "rowIndex": int(idx),
                            "question": question_text,
                            "answer": answer_text,
                            "numSources": len(context_chunks),
                        },
                    }
                ) + "\n"

        try:
            if file_ext == ".csv":
                df.to_csv(output_path, index=False)
            else:
                df.to_excel(output_path, index=False)
        except Exception as e:
            yield json.dumps({"type": "error", "data": {"message": f"Failed to write output file: {e}"}}) + "\n"
            return

        yield json.dumps({"type": "complete", "data": {"jobId": job_id, "processedRows": int(processed_rows)}}) + "\n"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/api/document_qa_download/{job_id}")
def document_qa_download(
    job_id: str,
    user: Dict[str, Any] = Depends(get_current_user_from_auth_header),
):
    meta = JOB_METADATA.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown jobId")

    if meta.get("organizationId") != user["organizationId"]:
        raise HTTPException(status_code=403, detail="Unauthorized")

    output_path = meta.get("outputPath")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Processed file not found")

    original_filename = meta.get("originalFilename") or "answered.xlsx"
    return FileResponse(
        output_path,
        filename=f"answered_{original_filename}",
    )


@app.get("/api/collections")
def get_collections(user: Dict[str, Any] = Depends(get_current_user_from_auth_header)):
    try:
        collections = db.get_user_collections(user["organizationId"])
        return {"status": "success", "data": collections}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/documents")
def get_documents(user: Dict[str, Any] = Depends(get_current_user_from_auth_header)):
    try:
        documents = db.get_organization_documents(user["organizationId"])
        return {"status": "success", "data": documents}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

