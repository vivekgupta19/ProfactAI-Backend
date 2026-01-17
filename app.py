from flask import Flask, request, jsonify, Response, stream_with_context, send_file
from flask_cors import CORS
from pathlib import Path
import os
import uuid
import pandas as pd

from config import TOP_K, validate_config
from rag import ingestion, vectorstore, llm
from auth import (
    register_user, login_user, logout_user, require_auth, 
    get_current_user_from_request, hash_password
)
from database import db

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PROCESSED_EXCEL_DIR = Path(
    os.environ.get("PROCESSED_EXCEL_OUTPUT_DIR", PROJECT_ROOT / "processed_excels")
)
PROCESSED_EXCEL_DIR.mkdir(parents=True, exist_ok=True)

JOB_METADATA = {}


def _get_cors_origins():
    """Return list of allowed CORS origins."""
    raw = os.environ.get("CORS_ORIGINS")
    if not raw:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app():
    app = Flask(__name__)
    
    # Enhanced CORS configuration
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": _get_cors_origins(),
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
                "expose_headers": ["Content-Type"],
                "supports_credentials": True,
                "max_age": 3600,
            }
        },
    )

    # ==================== HEALTH & SYSTEM ROUTES ====================
    
    @app.route("/api/health", methods=["GET"])
    @require_auth
    def health(user):
        try:
            # Check ChromaDB connection
            chroma_health = vectorstore.health_check()
            
            # Get collection stats
            collection = vectorstore.get_user_collection(user["organizationId"])
            count = collection.count()
            
            return jsonify({
                "status": "ok",
                "indexedRows": count,
                "organizationId": user["organizationId"],
                "organizationName": user["organizationName"],
                "chromaDB": chroma_health,
                "mode": "cloud" if os.environ.get("CHROMA_API_KEY") else "local"
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e),
                "organizationId": user["organizationId"]
            }), 500
    
    @app.route("/api/system/status", methods=["GET"])
    @require_auth
    def system_status(user):
        """Get detailed system status"""
        try:
            chroma_health = vectorstore.health_check()
            collections = vectorstore.list_all_collections()
            
            return jsonify({
                "status": "success",
                "data": {
                    "chromaDB": chroma_health,
                    "collections": collections,
                    "mode": "cloud" if os.environ.get("CHROMA_API_KEY") else "local"
                }
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500

    # ==================== AUTH ROUTES ====================
    
    @app.route("/api/auth/register", methods=["POST"])
    def register():
        if not request.is_json:
            return jsonify({"error": "Expected JSON body"}), 400
        
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")
        full_name = data.get("fullName")
        organization_name = data.get("organizationName")
        
        user_data, error = register_user(email, password, full_name, organization_name)
        
        if error:
            return jsonify({"error": error}), 400
        
        # Auto-login after registration
        auth_data, login_error = login_user(email, password)
        
        if login_error:
            return jsonify({"error": login_error}), 500
        
        return jsonify({
            "status": "success",
            "message": "Registration successful",
            "data": auth_data
        })
    
    @app.route("/api/auth/login", methods=["POST"])
    def login():
        if not request.is_json:
            return jsonify({"error": "Expected JSON body"}), 400
        
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")
        
        auth_data, error = login_user(email, password)
        
        if error:
            return jsonify({"error": error}), 401
        
        return jsonify({
            "status": "success",
            "data": auth_data
        })
    
    @app.route("/api/auth/logout", methods=["POST"])
    @require_auth
    def logout(user):
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            logout_user(token)
        
        return jsonify({"status": "success", "message": "Logged out successfully"})
    
    @app.route("/api/auth/me", methods=["GET"])
    @require_auth
    def get_current_user(user):
        return jsonify({"status": "success", "data": {"user": user}})

    # ==================== DOCUMENT MANAGEMENT ====================

    @app.route("/api/reindex", methods=["POST"])
    @require_auth
    def reindex(user):
        rebuild = request.json.get("rebuild", True) if request.is_json else True
        try:
            stats = ingestion.build_index(
                user["organizationId"],
                user["userId"],
                rebuild=rebuild
            )
            return jsonify({"status": "success", "stats": stats})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/upload_document", methods=["POST"])
    @require_auth
    def upload_document(user):
        """Upload a document and index it into ChromaDB"""
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        allowed_extensions = {".xlsx", ".xls", ".csv"}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                "error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            }), 400

        try:
            stats = ingestion.index_uploaded_file(
                file.stream,
                file.filename,
                user["organizationId"],
                user["userId"]
            )
            return jsonify({"status": "success", "stats": stats})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/search", methods=["POST"])
    @require_auth
    def search(user):
        if not request.is_json:
            return jsonify({"error": "Expected JSON body"}), 400
        
        data = request.get_json()
        query = data.get("query")
        top_k = int(data.get("top_k") or TOP_K)
        
        if not query:
            return jsonify({"error": "'query' is required"}), 400

        try:
            query_embedding = llm.embed_texts([query])[0]
            result = vectorstore.query_user_documents(
                user["organizationId"],
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            hits = []
            for i in range(len(result["ids"][0])):
                hits.append({
                    "id": result["ids"][0][i],
                    "document": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "distance": result.get("distances", [[None]])[0][i]
                    if result.get("distances") else None,
                })

            return jsonify({"query": query, "results": hits})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/chat", methods=["POST"])
    @require_auth
    def chat(user):
        if not request.is_json:
            return jsonify({"error": "Expected JSON body"}), 400
        
        data = request.get_json()
        question = data.get("question")
        top_k = int(data.get("top_k") or TOP_K)
        use_rag = bool(data.get("use_rag", True))

        if not question:
            return jsonify({"error": "'question' is required"}), 400

        try:
            context_chunks = []
            if use_rag:
                query_embedding = llm.embed_texts([question])[0]
                result = vectorstore.query_user_documents(
                    user["organizationId"],
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )

                for i in range(len(result["ids"][0])):
                    context_chunks.append({
                        "id": result["ids"][0][i],
                        "text": result["documents"][0][i],
                        "metadata": result["metadatas"][0][i],
                    })

            answer = llm.chat_with_context(question, context_chunks)

            return jsonify({
                "question": question,
                "answer": answer,
                "sources": context_chunks,
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/chat_stream", methods=["POST"])
    @require_auth
    def chat_stream(user):
        """Streaming version of /api/chat"""
        if not request.is_json:
            return jsonify({"error": "Expected JSON body"}), 400

        data = request.get_json()
        question = data.get("question")
        top_k = int(data.get("top_k") or TOP_K)
        use_rag = bool(data.get("use_rag", True))

        if not question:
            return jsonify({"error": "'question' is required"}), 400

        try:
            context_chunks = []
            if use_rag:
                query_embedding = llm.embed_texts([question])[0]
                result = vectorstore.query_user_documents(
                    user["organizationId"],
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )

                for i in range(len(result["ids"][0])):
                    context_chunks.append({
                        "id": result["ids"][0][i],
                        "text": result["documents"][0][i],
                        "metadata": result["metadatas"][0][i],
                    })

            def generate():
                import json

                header = {"question": question, "sources": context_chunks}
                yield json.dumps({"type": "meta", "data": header}) + "\n"

                for chunk in llm.chat_with_context_stream(question, context_chunks):
                    yield json.dumps({"type": "delta", "data": chunk}) + "\n"

            return Response(stream_with_context(generate()), mimetype="text/plain")
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/document_qa_stream", methods=["POST"])
    @require_auth
    def document_qa_stream(user):
        """Process document with Q&A"""
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        allowed_extensions = {".xlsx", ".xls", ".csv"}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                "error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            }), 400

        question_column = request.form.get("questionColumn")
        answer_column = request.form.get("answerColumn") or "aiAnswer"
        batch_size_raw = request.form.get("batchSize")
        max_rows_raw = request.form.get("maxRows")
        top_k_raw = request.form.get("topK")

        try:
            batch_size = int(batch_size_raw) if batch_size_raw else 10
            if batch_size <= 0:
                batch_size = 10
        except ValueError:
            batch_size = 10

        try:
            top_k = int(top_k_raw) if top_k_raw else TOP_K
            if top_k <= 0:
                top_k = TOP_K
        except ValueError:
            top_k = TOP_K

        max_rows = None
        if max_rows_raw:
            try:
                max_rows = int(max_rows_raw)
                if max_rows <= 0:
                    max_rows = None
            except ValueError:
                max_rows = None

        try:
            if file_ext == ".csv":
                df = pd.read_csv(file.stream)
            else:
                df = pd.read_excel(file.stream)
        except Exception as e:
            return jsonify({"error": f"Failed to read file: {e}"}), 400

        if df.empty:
            return jsonify({"error": "Uploaded file has no rows"}), 400

        if question_column and question_column in df.columns:
            q_col = question_column
        else:
            q_col = df.columns[0]
            question_column = q_col

        job_id = str(uuid.uuid4())
        output_path = PROCESSED_EXCEL_DIR / f"{job_id}{file_ext}"
        JOB_METADATA[job_id] = {
            "originalFilename": file.filename or f"uploaded{file_ext}",
            "outputPath": str(output_path),
            "questionColumn": question_column,
            "answerColumn": answer_column,
            "userId": user["userId"],
            "organizationId": user["organizationId"]
        }

        def generate():
            import json

            total_rows = len(df)
            limit = max_rows if max_rows is not None else total_rows
            if limit > total_rows:
                limit = total_rows

            header = {
                "jobId": job_id,
                "totalRows": int(total_rows),
                "limitRows": int(limit),
                "questionColumn": question_column,
                "answerColumn": answer_column,
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

                yield json.dumps({
                    "type": "status",
                    "data": {
                        "event": "batchStart",
                        "fromRow": int(start),
                        "toRow": int(end - 1),
                    },
                }) + "\n"

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
                            n_results=top_k
                        )

                        for i in range(len(result["ids"][0])):
                            context_chunks.append({
                                "id": result["ids"][0][i],
                                "text": result["documents"][0][i],
                                "metadata": result["metadatas"][0][i],
                            })
                    except Exception as e:
                        yield json.dumps({
                            "type": "warning",
                            "data": {
                                "rowIndex": int(idx),
                                "message": f"RAG retrieval failed: {e}",
                            },
                        }) + "\n"

                    try:
                        answer_text = llm.chat_with_context(question_text, context_chunks)
                    except Exception as e:
                        answer_text = f"Error generating answer: {e}"

                    df.loc[df.index[idx], answer_column] = answer_text
                    processed_rows += 1

                    yield json.dumps({
                        "type": "rowAnswer",
                        "data": {
                            "rowIndex": int(idx),
                            "question": question_text,
                            "answer": answer_text,
                            "numSources": len(context_chunks),
                        },
                    }) + "\n"

            try:
                if file_ext == ".csv":
                    df.to_csv(output_path, index=False)
                else:
                    df.to_excel(output_path, index=False)
            except Exception as e:
                error_msg = f"Failed to write output file: {e}"
                yield json.dumps({"type": "error", "data": {"message": error_msg}}) + "\n"
                return

            yield json.dumps({
                "type": "complete",
                "data": {
                    "jobId": job_id,
                    "processedRows": int(processed_rows),
                },
            }) + "\n"

        return Response(stream_with_context(generate()), mimetype="text/plain")

    @app.route("/api/document_qa_download/<job_id>", methods=["GET"])
    @require_auth
    def document_qa_download(job_id, user):
        meta = JOB_METADATA.get(job_id)
        if not meta:
            return jsonify({"error": "Unknown jobId"}), 404

        if meta.get("organizationId") != user["organizationId"]:
            return jsonify({"error": "Unauthorized"}), 403

        output_path = meta.get("outputPath")
        if not output_path or not os.path.exists(output_path):
            return jsonify({"error": "Processed file not found"}), 404

        original_filename = meta.get("originalFilename") or "answered.xlsx"

        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"answered_{original_filename}",
        )
    
    @app.route("/api/collections", methods=["GET"])
    @require_auth
    def get_collections(user):
        """Get all collections for the user's organization"""
        try:
            collections = db.get_user_collections(user["organizationId"])
            return jsonify({"status": "success", "data": collections})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/api/documents", methods=["GET"])
    @require_auth
    def get_documents(user):
        """Get all documents for the user's organization"""
        try:
            documents = db.get_organization_documents(user["organizationId"])
            return jsonify({"status": "success", "data": documents})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return app


if __name__ == "__main__":
    # Validate configuration
    validate_config()
    
    # Initialize superadmin if not exists
    try:
        existing = db.get_user_by_email("vivekgupta@profact.ai")
        if not existing:
            org = db.get_organization_by_slug("profact-admin")
            if org:
                password_hash = hash_password("vivek123")
                db.create_user(
                    email="vivekgupta@profact.ai",
                    password_hash=password_hash,
                    full_name="Vivek Gupta",
                    organization_id=org["organizationid"],
                    role="superadmin"
                )
                print("✅ Superadmin user created successfully")
    except Exception as e:
        print(f"⚠️  Could not create superadmin: {e}")
    
    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=5001, debug=True)