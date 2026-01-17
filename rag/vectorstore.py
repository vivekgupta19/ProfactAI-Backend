from __future__ import annotations

from dataclasses import dataclass
import json
from typing import List, Optional, Dict, Any

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json

from config import PGVECTOR_TABLE, PGVECTOR_DIM
from database import db

_schema_ready = False


def _embedding_to_pgvector_literal(embedding: List[float]) -> str:
    # pgvector accepts: '[1,2,3]'::vector
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def ensure_schema() -> None:
    """Ensure pgvector extension + tables exist in the configured DATABASE_URL."""
    global _schema_ready
    if _schema_ready:
        return

    table = PGVECTOR_TABLE
    dim = int(PGVECTOR_DIM)

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                      id TEXT PRIMARY KEY,
                      organization_id TEXT NOT NULL,
                      document TEXT NOT NULL,
                      metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                      embedding VECTOR({dim}) NOT NULL,
                      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                ).format(table=sql.Identifier(table), dim=sql.Literal(dim))
            )

            # Helpful indexes (best-effort; don't fail app startup if they can't be created)
            try:
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} (organization_id);").format(
                        idx=sql.Identifier(f"{table}_org_idx"),
                        table=sql.Identifier(table),
                    )
                )
            except Exception:
                pass

            # Vector index (ivfflat). Works well at scale; requires ANALYZE for best results.
            try:
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {idx} ON {table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
                    ).format(
                        idx=sql.Identifier(f"{table}_embedding_ivfflat_idx"),
                        table=sql.Identifier(table),
                    )
                )
            except Exception:
                pass

    _schema_ready = True


def get_collection_name(organization_id: str) -> str:
    """Keep a stable 'collection name' for compatibility with previous Chroma naming."""
    safe_org_id = organization_id.replace("-", "_")
    return f"org_{safe_org_id}"


@dataclass(frozen=True)
class _UserCollection:
    organization_id: str

    @property
    def name(self) -> str:
        return get_collection_name(self.organization_id)

    def count(self) -> int:
        ensure_schema()
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {table} WHERE organization_id = %s;").format(
                        table=sql.Identifier(PGVECTOR_TABLE)
                    ),
                    (self.organization_id,),
                )
                return int(cur.fetchone()[0])


def get_user_collection(organization_id: str) -> _UserCollection:
    ensure_schema()
    return _UserCollection(organization_id=organization_id)


def reset_user_collection(organization_id: str) -> _UserCollection:
    """Delete all vectors for an organization (equivalent to dropping the org's Chroma collection)."""
    ensure_schema()
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {table} WHERE organization_id = %s;").format(
                    table=sql.Identifier(PGVECTOR_TABLE)
                ),
                (organization_id,),
            )
    return _UserCollection(organization_id=organization_id)


def add_documents(
    organization_id: str,
    texts: List[str],
    metadatas: List[dict],
    ids: List[str],
    embeddings: Optional[List[List[float]]] = None,
):
    """Upsert documents into pgvector store."""
    ensure_schema()
    if embeddings is None:
        raise ValueError("embeddings are required (this backend stores precomputed vectors).")
    if not (len(texts) == len(metadatas) == len(ids) == len(embeddings)):
        raise ValueError("texts, metadatas, ids, embeddings must have the same length")

    rows = []
    for doc_id, text, meta, emb in zip(ids, texts, metadatas, embeddings):
        vec = _embedding_to_pgvector_literal(emb)
        rows.append((doc_id, organization_id, text, Json(meta), vec))

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {table} (id, organization_id, document, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                      organization_id = EXCLUDED.organization_id,
                      document = EXCLUDED.document,
                      metadata = EXCLUDED.metadata,
                      embedding = EXCLUDED.embedding;
                    """
                ).format(table=sql.Identifier(PGVECTOR_TABLE)),
                rows,
            )


def query_user_documents(
    organization_id: str,
    query_texts: Optional[List[str]] = None,
    query_embeddings: Optional[List[List[float]]] = None,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
):
    """Query documents in an organization's collection using cosine distance."""
    ensure_schema()
    if query_embeddings is None and query_texts is None:
        raise ValueError("Either query_texts or query_embeddings must be provided")
    if query_embeddings is None:
        raise ValueError("query_texts is not supported here; pass query_embeddings from llm.embed_texts()")
    if len(query_embeddings) != 1:
        raise ValueError("Only single-query retrieval is supported (pass a single query embedding).")

    vec = _embedding_to_pgvector_literal(query_embeddings[0])

    filters_sql = sql.SQL("")
    params: list[Any] = [organization_id]

    if where:
        # Support simple metadata equality filters: {"file": "x.xlsx"}
        clauses = []
        for k, v in where.items():
            clauses.append(sql.SQL("(metadata ->> %s) = %s"))
            params.extend([str(k), str(v)])
        filters_sql = sql.SQL(" AND ") + sql.SQL(" AND ").join(clauses)

    query = sql.SQL(
        """
        SELECT
          id,
          document,
          metadata::text AS metadata,
          (embedding <=> %s::vector) AS distance
        FROM {table}
        WHERE organization_id = %s
        {filters}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
    ).format(table=sql.Identifier(PGVECTOR_TABLE), filters=filters_sql)

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (vec, *params, vec, int(n_results)))
            rows = cur.fetchall()

    ids_out: List[str] = []
    docs_out: List[str] = []
    metas_out: List[dict] = []
    dists_out: List[float] = []

    for row in rows:
        # cursor is not RealDictCursor here; row = tuple
        doc_id, doc, meta_text, dist = row
        ids_out.append(doc_id)
        docs_out.append(doc)
        metas_out.append(json.loads(meta_text) if meta_text else {})
        dists_out.append(float(dist) if dist is not None else None)

    return {
        "ids": [ids_out],
        "documents": [docs_out],
        "metadatas": [metas_out],
        "distances": [dists_out],
    }


def get_collection_stats(organization_id: str) -> Dict[str, Any]:
    """Get statistics about an organization's collection."""
    try:
        collection = get_user_collection(organization_id)
        return {"name": collection.name, "count": collection.count(), "organizationId": organization_id}
    except Exception as e:
        return {"name": None, "count": 0, "organizationId": organization_id, "error": str(e)}