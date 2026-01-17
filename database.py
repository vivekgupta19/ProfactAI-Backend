import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.connection_string = os.environ.get("DATABASE_URL")
        if not self.connection_string:
            raise RuntimeError("DATABASE_URL environment variable is not set")

    @contextmanager
    def get_connection(self):
        conn = psycopg2.connect(self.connection_string)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = False
    ) -> Optional[Any]:
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params or ())
                if fetch_one:
                    return dict(cursor.fetchone()) if cursor.rowcount > 0 else None
                elif fetch_all:
                    return [dict(row) for row in cursor.fetchall()]
                return cursor.rowcount

    def create_user(
        self,
        email: str,
        password_hash: str,
        full_name: str,
        organization_id: str,
        role: str = "user"
    ) -> Dict[str, Any]:
        query = """
            INSERT INTO users (email, passwordHash, fullName, organizationId, role)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING userId, email, fullName, organizationId, role, createdAt
        """
        return self.execute_query(
            query,
            (email, password_hash, full_name, organization_id, role),
            fetch_one=True
        )

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        query = """
            SELECT u.userId, u.email, u.passwordHash, u.fullName, u.role, 
                   u.organizationId, u.isActive, u.createdAt,
                   o.organizationName, o.organizationSlug
            FROM users u
            JOIN organizations o ON u.organizationId = o.organizationId
            WHERE u.email = %s AND u.isActive = TRUE
        """
        return self.execute_query(query, (email,), fetch_one=True)

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        query = """
            SELECT u.userId, u.email, u.fullName, u.role, 
                   u.organizationId, u.isActive, u.createdAt,
                   o.organizationName, o.organizationSlug
            FROM users u
            JOIN organizations o ON u.organizationId = o.organizationId
            WHERE u.userId = %s AND u.isActive = TRUE
        """
        return self.execute_query(query, (user_id,), fetch_one=True)

    def update_last_login(self, user_id: str):
        query = "UPDATE users SET lastLoginAt = NOW() WHERE userId = %s"
        self.execute_query(query, (user_id,))

    def create_session(self, user_id: str, token: str, expires_at) -> Dict[str, Any]:
        query = """
            INSERT INTO user_sessions (userId, token, expiresAt)
            VALUES (%s, %s, %s)
            RETURNING sessionId, userId, token, expiresAt, createdAt
        """
        return self.execute_query(
            query,
            (user_id, token, expires_at),
            fetch_one=True
        )

    def get_session_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        query = """
            SELECT s.sessionId, s.userId, s.token, s.expiresAt,
                   u.email, u.fullName, u.role, u.organizationId,
                   o.organizationName, o.organizationSlug
            FROM user_sessions s
            JOIN users u ON s.userId = u.userId
            JOIN organizations o ON u.organizationId = o.organizationId
            WHERE s.token = %s AND s.expiresAt > NOW() AND u.isActive = TRUE
        """
        return self.execute_query(query, (token,), fetch_one=True)

    def delete_session(self, token: str):
        query = "DELETE FROM user_sessions WHERE token = %s"
        self.execute_query(query, (token,))

    def create_organization(self, organization_name: str, organization_slug: str) -> Dict[str, Any]:
        query = """
            INSERT INTO organizations (organizationName, organizationSlug)
            VALUES (%s, %s)
            RETURNING organizationId, organizationName, organizationSlug, createdAt
        """
        return self.execute_query(
            query,
            (organization_name, organization_slug),
            fetch_one=True
        )

    def get_organization_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        query = """
            SELECT organizationId, organizationName, organizationSlug, createdAt
            FROM organizations
            WHERE organizationSlug = %s
        """
        return self.execute_query(query, (slug,), fetch_one=True)

    def create_collection(
        self,
        organization_id: str,
        user_id: str,
        collection_name: str,
        chroma_collection_name: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        query = """
            INSERT INTO collections 
            (organizationId, userId, collectionName, chromaCollectionName, description)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING collectionId, organizationId, userId, collectionName, 
                      chromaCollectionName, description, createdAt
        """
        return self.execute_query(
            query,
            (organization_id, user_id, collection_name, chroma_collection_name, description),
            fetch_one=True
        )

    def get_user_collections(self, organization_id: str) -> List[Dict[str, Any]]:
        query = """
            SELECT c.collectionId, c.organizationId, c.userId, c.collectionName,
                   c.chromaCollectionName, c.description, c.createdAt,
                   u.fullName as createdByName
            FROM collections c
            JOIN users u ON c.userId = u.userId
            WHERE c.organizationId = %s
            ORDER BY c.createdAt DESC
        """
        return self.execute_query(query, (organization_id,), fetch_all=True)

    def get_collection_by_id(self, collection_id: str) -> Optional[Dict[str, Any]]:
        query = """
            SELECT collectionId, organizationId, userId, collectionName,
                   chromaCollectionName, description, createdAt
            FROM collections
            WHERE collectionId = %s
        """
        return self.execute_query(query, (collection_id,), fetch_one=True)

    def create_document(
        self,
        collection_id: str,
        organization_id: str,
        user_id: str,
        file_name: str,
        file_type: str,
        file_size: int,
        total_rows: int,
        indexed_rows: int
    ) -> Dict[str, Any]:
        query = """
            INSERT INTO documents 
            (collectionId, organizationId, userId, fileName, fileType, 
             fileSize, totalRows, indexedRows)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING documentId, collectionId, organizationId, userId, fileName,
                      fileType, fileSize, totalRows, indexedRows, createdAt
        """
        return self.execute_query(
            query,
            (collection_id, organization_id, user_id, file_name, file_type,
             file_size, total_rows, indexed_rows),
            fetch_one=True
        )

    def get_organization_documents(self, organization_id: str) -> List[Dict[str, Any]]:
        query = """
            SELECT d.documentId, d.fileName, d.fileType, d.fileSize, 
                   d.totalRows, d.indexedRows, d.createdAt,
                   c.collectionName, u.fullName as uploadedByName
            FROM documents d
            JOIN collections c ON d.collectionId = c.collectionId
            JOIN users u ON d.userId = u.userId
            WHERE d.organizationId = %s
            ORDER BY d.createdAt DESC
        """
        return self.execute_query(query, (organization_id,), fetch_all=True)


# Global database instance
db = Database()