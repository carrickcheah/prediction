"""
Database connection configuration and management.
"""
import mysql.connector
from mysql.connector import Error, pooling
from contextlib import contextmanager
import logging
from typing import Optional, Dict, Any, List

from .settings import get_settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections with connection pooling."""
    
    def __init__(self):
        self.settings = get_settings()
        self._pool: Optional[pooling.MySQLConnectionPool] = None
    
    def _create_pool(self) -> pooling.MySQLConnectionPool:
        """Create a connection pool."""
        config = {
            "host": self.settings.MARIADB_HOST,
            "port": self.settings.MARIADB_PORT,
            "database": self.settings.MARIADB_DATABASE,
            "user": self.settings.MARIADB_USER,
            "password": self.settings.MARIADB_PASSWORD,
            "pool_name": "inventory_pool",
            "pool_size": 5,
            "pool_reset_session": True,
        }
        
        return pooling.MySQLConnectionPool(**config)
    
    @property
    def pool(self) -> pooling.MySQLConnectionPool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = self._create_pool()
        return self._pool
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        connection = None
        try:
            connection = self.pool.get_connection()
            yield connection
        except Error as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    def execute_query(self, query: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as list of dicts."""
        with self.get_connection() as connection:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(query, params or [])
                return cursor.fetchall()
            finally:
                cursor.close()
    
    def execute_update(self, query: str, params: Optional[List] = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        with self.get_connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(query, params or [])
                connection.commit()
                return cursor.rowcount
            except Exception as e:
                connection.rollback()
                raise e
            finally:
                cursor.close()


# Singleton instance
db_manager = DatabaseManager()