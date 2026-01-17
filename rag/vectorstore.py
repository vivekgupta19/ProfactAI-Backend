from typing import List, Optional, Dict, Any
import chromadb
from config import CHROMA_DB_DIR

_client: Optional[chromadb.PersistentClient] = None
_collections_cache: Dict[str, any] = {}


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _client


def get_collection_name(organization_id: str) -> str:
    """Generate a unique collection name for an organization"""
    # Replace hyphens with underscores for ChromaDB compatibility
    safe_org_id = organization_id.replace("-", "_")
    return f"org_{safe_org_id}"


def get_user_collection(organization_id: str):
    """Get or create a collection for a specific organization"""
    global _collections_cache
    
    collection_name = get_collection_name(organization_id)
    
    if collection_name in _collections_cache:
        return _collections_cache[collection_name]
    
    client = get_client()
    collection = client.get_or_create_collection(collection_name)
    _collections_cache[collection_name] = collection
    
    return collection


def reset_user_collection(organization_id: str):
    """Reset (delete and recreate) a collection for an organization"""
    global _collections_cache
    
    collection_name = get_collection_name(organization_id)
    client = get_client()
    
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection might not exist
    
    collection = client.create_collection(collection_name)
    _collections_cache[collection_name] = collection
    
    return collection


def add_documents(
    organization_id: str,
    texts: List[str],
    metadatas: List[dict],
    ids: List[str],
    embeddings: Optional[List[List[float]]] = None,
):
    """Add documents to an organization's collection"""
    collection = get_user_collection(organization_id)
    kwargs = {"documents": texts, "metadatas": metadatas, "ids": ids}
    if embeddings is not None:
        kwargs["embeddings"] = embeddings
    collection.add(**kwargs)


def query_user_documents(
    organization_id: str,
    query_texts: Optional[List[str]] = None,
    query_embeddings: Optional[List[List[float]]] = None,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
):
    """Query documents in an organization's collection"""
    collection = get_user_collection(organization_id)
    query_kwargs: Dict[str, Any] = {"n_results": n_results}
    
    if where is not None:
        query_kwargs["where"] = where

    if query_embeddings is not None:
        query_kwargs["query_embeddings"] = query_embeddings
        return collection.query(**query_kwargs)
    if query_texts is not None:
        query_kwargs["query_texts"] = query_texts
        return collection.query(**query_kwargs)
    
    raise ValueError("Either query_texts or query_embeddings must be provided")


def get_collection_stats(organization_id: str) -> Dict[str, Any]:
    """Get statistics about an organization's collection"""
    try:
        collection = get_user_collection(organization_id)
        return {
            "name": collection.name,
            "count": collection.count(),
            "organizationId": organization_id
        }
    except Exception as e:
        return {
            "name": None,
            "count": 0,
            "organizationId": organization_id,
            "error": str(e)
        }