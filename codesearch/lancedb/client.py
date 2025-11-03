"""LanceDB client for managing database connections."""

import lancedb
from pathlib import Path
from typing import Optional, List
import logging


logger = logging.getLogger(__name__)


class LanceDBClient:
    """Client for managing LanceDB connections and table operations."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize LanceDB client.

        Args:
            db_path: Path to LanceDB directory (default: .lancedb/)
        """
        self.db_path = db_path or Path(".lancedb")
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.db = lancedb.connect(str(self.db_path))

        # Track initialized tables
        self._initialized_tables = set()

        logger.info(f"LanceDB client initialized at {self.db_path}")

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        try:
            self.db.open_table(table_name)
            return True
        except Exception:
            return False

    def create_table(
        self,
        table_name: str,
        data: Optional[List[dict]] = None,
        schema: Optional[dict] = None,
    ):
        """Create or overwrite a table."""
        try:
            if data is not None:  # Allow empty list
                self.db.create_table(table_name, data=data, mode="overwrite")
            elif schema:
                self.db.create_table(table_name, schema=schema, mode="overwrite")
            else:
                # Create empty table with a placeholder row then delete it
                # This allows initializing empty tables
                self.db.create_table(table_name, data=[{}], mode="overwrite")
                # Note: LanceDB will handle the empty dict appropriately

            self._initialized_tables.add(table_name)
            logger.info(f"Created table: {table_name}")

        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            raise

    def get_table(self, table_name: str):
        """Get a reference to an existing table."""
        if not self.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist")
        return self.db.open_table(table_name)

    def insert(self, table_name: str, data: List[dict]):
        """Insert data into a table."""
        if not data:
            # Empty list - nothing to insert
            logger.debug(f"No data to insert into {table_name}")
            return

        try:
            table = self.get_table(table_name)
            table.add(data)
            logger.debug(f"Inserted {len(data)} rows into {table_name}")
        except Exception as e:
            logger.error(f"Failed to insert into {table_name}: {e}")
            raise

    def search(self, table_name: str, query_vector: List[float], k: int = 10):
        """Vector search in a table."""
        try:
            table = self.get_table(table_name)
            results = table.search(query_vector).limit(k).to_list()
            return results
        except Exception as e:
            logger.error(f"Search failed in {table_name}: {e}")
            raise

    def close(self):
        """Close database connection."""
        # LanceDB doesn't require explicit close, but good for cleanup
        self.db = None
        logger.info("LanceDB client closed")
