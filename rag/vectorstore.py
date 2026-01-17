from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
import os

_client: Optional[chromadb.ClientAPI] = None
_collections_cache: Dict[str, any] = {}


def get_client() -> chromadb.ClientAPI:
    """Get or create ChromaDB client (cloud-based)"""
    global _client
    if _client is None:
        # Cloud ChromaDB configuration
        chroma_api_key = os.environ.get("CHROMA_API_KEY")
        chroma_tenant = os.environ.get("CHROMA_TENANT")
        chroma_database = os.environ.get("CHROMA_DATABASE")
        
        if chroma_api_key and chroma_tenant and chroma_database:
            # Chroma Cloud mode (official managed service)
            _client = chromadb.CloudClient(
                api_key=chroma_api_key,
                tenant=chroma_tenant,
                database=chroma_database
            )
            print(f"✅ Connected to Chroma Cloud - Tenant: {chroma_tenant}, Database: {chroma_database}")
        else:
            # Fallback to local persistent client
            from config import CHROMA_DB_DIR
            _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            print(f"⚠️  Using local ChromaDB at {CHROMA_DB_DIR}")
            print("   Set CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE to use Chroma Cloud")
    
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
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"organizationId": organization_id}
    )
    _collections_cache[collection_name] = collection
    
    return collection


def reset_user_collection(organization_id: str):
    """Reset (delete and recreate) a collection for an organization"""
    global _collections_cache
    
    collection_name = get_collection_name(organization_id)
    client = get_client()
    
    try:
        client.delete_collection(collection_name)
        print(f"🗑️  Deleted collection: {collection_name}")
    except Exception as e:
        print(f"ℹ️  Collection {collection_name} does not exist: {e}")
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"organizationId": organization_id}
    )
    _collections_cache[collection_name] = collection
    print(f"✅ Created new collection: {collection_name}")
    
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
    
    try:
        collection.add(**kwargs)
    except Exception as e:
        # Handle duplicate IDs by upserting instead
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            collection.upsert(**kwargs)
        else:
            raise e


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
        count = collection.count()
        return {
            "name": collection.name,
            "count": count,
            "organizationId": organization_id,
            "metadata": collection.metadata
        }
    except Exception as e:
        return {
            "name": None,
            "count": 0,
            "organizationId": organization_id,
            "error": str(e)
        }


def delete_documents_by_file(organization_id: str, file_name: str) -> int:
    """Delete all documents associated with a specific file"""
    try:
        collection = get_user_collection(organization_id)
        
        # Query for all documents from this file
        results = collection.get(
            where={"file": file_name}
        )
        
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            return len(results["ids"])
        
        return 0
    except Exception as e:
        print(f"Error deleting documents for file {file_name}: {e}")
        return 0


def list_all_collections() -> List[Dict[str, Any]]:
    """List all collections in the ChromaDB instance"""
    try:
        client = get_client()
        collections = client.list_collections()
        return [
            {
                "name": col.name,
                "count": col.count(),
                "metadata": col.metadata
            }
            for col in collections
        ]
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []


def health_check() -> Dict[str, Any]:
    """Check ChromaDB connection health"""
    try:
        client = get_client()
        # Try to get heartbeat (if available)
        heartbeat = client.heartbeat()
        return {
            "status": "healthy",
            "heartbeat": heartbeat,
            "collections_count": len(client.list_collections())
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }