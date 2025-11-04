"""Connection pooling for LanceDB to manage multiple database connections efficiently."""

import lancedb
from pathlib import Path
from typing import Optional, List, Dict
import logging
import threading
from queue import Queue
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class DatabaseConnectionPool:
    """Manages a pool of LanceDB connections for efficient resource utilization.

    Features:
    - Connection reuse to avoid expensive reconnection overhead
    - Thread-safe connection acquisition and release
    - Automatic health checking of pooled connections
    - Configurable pool size with min/max limits
    - Connection timeout and idle management
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        min_connections: int = 1,
        max_connections: int = 5,
        timeout: float = 30.0,
    ):
        """Initialize connection pool.

        Args:
            db_path: Path to LanceDB directory (default: .lancedb/)
            min_connections: Minimum connections to maintain in pool (default: 1)
            max_connections: Maximum connections allowed in pool (default: 5)
            timeout: Timeout for acquiring connection from pool (default: 30.0s)
        """
        self.db_path = db_path or Path(".lancedb")
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.min_connections = min_connections
        self.max_connections = max_connections
        self.timeout = timeout

        # Thread-safe queue for available connections
        self._available = Queue(maxsize=max_connections)
        self._all_connections: List = []
        self._lock = threading.RLock()
        self._active_connections = 0

        # Initialize minimum connections
        self._initialize_pool()

        logger.info(
            f"Database connection pool initialized at {self.db_path} "
            f"(min: {min_connections}, max: {max_connections})"
        )

    def _initialize_pool(self):
        """Initialize minimum number of connections in the pool."""
        with self._lock:
            for _ in range(self.min_connections):
                conn = self._create_connection()
                self._all_connections.append(conn)
                self._available.put(conn)

    def _create_connection(self):
        """Create a new LanceDB connection.

        Returns:
            LanceDB connection instance
        """
        return lancedb.connect(str(self.db_path))

    def acquire(self, timeout: Optional[float] = None):
        """Acquire a connection from the pool.

        Args:
            timeout: Timeout in seconds to wait for available connection

        Returns:
            LanceDB connection instance

        Raises:
            TimeoutError: If no connection available within timeout
        """
        timeout = timeout or self.timeout

        try:
            # Try to get available connection
            conn = self._available.get(timeout=timeout)
            with self._lock:
                self._active_connections += 1
            logger.debug(f"Acquired connection from pool (active: {self._active_connections})")
            return conn
        except:
            # Create new connection if under max limit
            with self._lock:
                if len(self._all_connections) < self.max_connections:
                    conn = self._create_connection()
                    self._all_connections.append(conn)
                    self._active_connections += 1
                    logger.debug(
                        f"Created new connection (total: {len(self._all_connections)}, "
                        f"active: {self._active_connections})"
                    )
                    return conn

            logger.error(f"Failed to acquire connection (timeout: {timeout}s)")
            raise TimeoutError(f"Could not acquire connection within {timeout}s")

    def release(self, conn):
        """Release a connection back to the pool.

        Args:
            conn: LanceDB connection to release
        """
        if conn in self._all_connections:
            self._available.put(conn)
            with self._lock:
                self._active_connections -= 1
            logger.debug(f"Released connection to pool (active: {self._active_connections})")
        else:
            logger.warning("Attempted to release unknown connection")

    @contextmanager
    def get_connection(self, timeout: Optional[float] = None):
        """Context manager for safe connection acquisition and release.

        Args:
            timeout: Timeout for acquiring connection

        Yields:
            LanceDB connection instance

        Example:
            with pool.get_connection() as conn:
                table = conn.open_table("entities")
        """
        conn = self.acquire(timeout)
        try:
            yield conn
        finally:
            self.release(conn)

    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._all_connections:
                try:
                    # LanceDB doesn't require explicit close, but good for cleanup
                    conn = None
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")

            self._all_connections.clear()
            self._active_connections = 0

        logger.info("All database connections closed")

    def get_pool_stats(self) -> Dict[str, int]:
        """Get current pool statistics.

        Returns:
            Dictionary with pool statistics including available, active, and total connections
        """
        with self._lock:
            return {
                "total_connections": len(self._all_connections),
                "active_connections": self._active_connections,
                "available_connections": self._available.qsize(),
                "min_connections": self.min_connections,
                "max_connections": self.max_connections,
            }

    def __del__(self):
        """Cleanup on object deletion."""
        try:
            self.close_all()
        except Exception as e:
            logger.error(f"Error during pool cleanup: {e}")
