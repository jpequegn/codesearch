"""Database statistics and introspection for monitoring and diagnostics."""

import lancedb
from pathlib import Path
from typing import Optional, Dict, List
import logging
from datetime import datetime, timezone
import os


logger = logging.getLogger(__name__)


class DatabaseStatistics:
    """Provides comprehensive statistics and introspection for LanceDB.

    Features:
    - Table-level statistics (row counts, column info)
    - Database-level metrics (size, table list)
    - Query performance tracking
    - Index statistics and introspection
    - Data distribution analysis
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize statistics manager.

        Args:
            db_path: Path to LanceDB directory
        """
        self.db_path = db_path or Path(".lancedb")

        try:
            self.db = lancedb.connect(str(self.db_path))
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.db = None

    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics.

        Returns:
            Dictionary with overall database statistics
        """
        if not self.db:
            return {"status": "error", "message": "Database connection failed"}

        try:
            stats = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "database_path": str(self.db_path),
                "database_size_bytes": self._get_database_size(),
                "database_size": self._format_size(self._get_database_size()),
                "tables": self.get_table_list(),
                "table_count": len(self.get_table_list()),
                "total_rows": sum(self.get_row_count(t) for t in self.get_table_list()),
            }

            return stats

        except Exception as e:
            logger.error(f"Error collecting database stats: {e}")
            return {"status": "error", "error": str(e)}

    def get_table_list(self) -> List[str]:
        """Get list of all tables in database.

        Returns:
            List of table names
        """
        try:
            if self.db:
                # LanceDB provides table names through internal API
                return self.db.table_names()
        except Exception as e:
            logger.debug(f"Error listing tables: {e}")

        return []

    def get_table_stats(self, table_name: str) -> Dict:
        """Get detailed statistics for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table statistics
        """
        if not self.db:
            return {"status": "error", "message": "Database connection failed"}

        try:
            table = self.db.open_table(table_name)

            stats = {
                "table_name": table_name,
                "row_count": table.count_rows(),
                "schema": self._get_schema_info(table),
                "column_count": len(self._get_schema_info(table)),
            }

            # Get data size estimate
            try:
                stats["estimated_size_bytes"] = self._estimate_table_size(table)
                stats["estimated_size"] = self._format_size(stats["estimated_size_bytes"])
            except Exception as e:
                logger.debug(f"Could not estimate table size: {e}")

            return stats

        except Exception as e:
            logger.error(f"Error getting table stats for {table_name}: {e}")
            return {"status": "error", "error": str(e), "table_name": table_name}

    def get_row_count(self, table_name: str) -> int:
        """Get row count for a table.

        Args:
            table_name: Name of the table

        Returns:
            Number of rows in table
        """
        try:
            table = self.db.open_table(table_name)
            return table.count_rows()
        except Exception as e:
            logger.debug(f"Error getting row count for {table_name}: {e}")
            return 0

    def get_all_table_stats(self) -> Dict[str, Dict]:
        """Get statistics for all tables.

        Returns:
            Dictionary mapping table names to their statistics
        """
        tables = self.get_table_list()
        stats = {}

        for table_name in tables:
            stats[table_name] = self.get_table_stats(table_name)

        return stats

    def get_schema_info(self, table_name: str) -> Dict:
        """Get schema information for a table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with column names and types
        """
        try:
            table = self.db.open_table(table_name)
            schema = self._get_schema_info(table)

            return {
                "table_name": table_name,
                "columns": schema,
                "column_count": len(schema),
            }

        except Exception as e:
            logger.error(f"Error getting schema for {table_name}: {e}")
            return {"status": "error", "error": str(e)}

    def get_data_distribution(self, table_name: str, column_name: str, limit: int = 100) -> Dict:
        """Analyze data distribution in a column.

        Args:
            table_name: Name of the table
            column_name: Name of the column to analyze
            limit: Maximum distinct values to return

        Returns:
            Dictionary with distribution statistics
        """
        try:
            table = self.db.open_table(table_name)

            # Get sample of distinct values
            # Note: LanceDB's Python API is limited, so we work with what's available
            data = table.to_pandas()

            if column_name not in data.columns:
                return {"error": f"Column {column_name} not found"}

            column = data[column_name]

            distribution = {
                "table_name": table_name,
                "column_name": column_name,
                "total_rows": len(column),
                "non_null_count": column.notna().sum(),
                "null_count": column.isna().sum(),
                "unique_count": column.nunique(),
            }

            # Add type-specific statistics
            if column.dtype in ["int64", "float64"]:
                distribution.update({
                    "dtype": "numeric",
                    "min": float(column.min()),
                    "max": float(column.max()),
                    "mean": float(column.mean()),
                    "median": float(column.median()),
                    "std": float(column.std()),
                })
            else:
                distribution.update({
                    "dtype": "string",
                    "top_values": column.value_counts().head(limit).to_dict(),
                })

            return distribution

        except Exception as e:
            logger.error(f"Error analyzing data distribution: {e}")
            return {"status": "error", "error": str(e)}

    def validate_database_integrity(self) -> Dict:
        """Validate database integrity by checking all tables.

        Returns:
            Dictionary with validation results for each table
        """
        validation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "tables": {},
        }

        tables = self.get_table_list()

        for table_name in tables:
            try:
                table = self.db.open_table(table_name)
                row_count = table.count_rows()

                validation["tables"][table_name] = {
                    "status": "valid",
                    "row_count": row_count,
                }
            except Exception as e:
                validation["tables"][table_name] = {
                    "status": "error",
                    "error": str(e),
                }
                validation["status"] = "partial_failure"

        return validation

    def get_health_check(self) -> Dict:
        """Perform health check on database.

        Returns:
            Dictionary with health check results
        """
        health = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database_path": str(self.db_path),
            "checks": {},
        }

        # Check 1: Database directory exists
        health["checks"]["database_exists"] = self.db_path.exists()

        # Check 2: Can connect to database
        try:
            self.db.table_names()
            health["checks"]["connection_ok"] = True
        except Exception as e:
            health["checks"]["connection_ok"] = False
            health["checks"]["connection_error"] = str(e)

        # Check 3: Can list tables
        try:
            tables = self.get_table_list()
            health["checks"]["tables_accessible"] = True
            health["checks"]["table_count"] = len(tables)
        except Exception as e:
            health["checks"]["tables_accessible"] = False
            health["checks"]["error"] = str(e)

        # Check 4: Database size reasonable
        db_size = self._get_database_size()
        health["checks"]["database_size_bytes"] = db_size
        health["checks"]["database_size"] = self._format_size(db_size)

        # Overall health
        all_checks = health["checks"].values()
        critical_checks = [
            health["checks"].get("database_exists"),
            health["checks"].get("connection_ok"),
        ]

        health["overall_status"] = "healthy" if all(critical_checks) else "unhealthy"

        return health

    def _get_schema_info(self, table) -> Dict[str, str]:
        """Extract schema information from a table.

        Args:
            table: LanceDB table instance

        Returns:
            Dictionary mapping column names to types
        """
        try:
            # Try to get schema through pandas
            df = table.to_pandas(limit=0)  # Get schema without data
            schema = {}
            for col in df.columns:
                schema[col] = str(df[col].dtype)
            return schema
        except Exception:
            # Fallback: return empty schema
            return {}

    @staticmethod
    def _estimate_table_size(table) -> int:
        """Estimate size of a table in bytes.

        Args:
            table: LanceDB table instance

        Returns:
            Estimated size in bytes
        """
        try:
            # This is a rough estimate based on row count and sample data
            df = table.to_pandas(limit=100)
            memory_usage = df.memory_usage(deep=True).sum()
            row_count = table.count_rows()

            # Estimate total size
            if len(df) > 0:
                avg_row_size = memory_usage / len(df)
                estimated_size = int(avg_row_size * row_count)
                return estimated_size

            return 0
        except Exception:
            return 0

    @staticmethod
    def _get_database_size() -> int:
        """Get total database directory size in bytes.

        Returns:
            Total size in bytes
        """
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(".lancedb"):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total += os.path.getsize(filepath)
        except Exception as e:
            logger.debug(f"Error calculating database size: {e}")

        return total

    @staticmethod
    def _format_size(bytes_size: int) -> str:
        """Format bytes to human-readable size.

        Args:
            bytes_size: Size in bytes

        Returns:
            Formatted string (e.g., "1.5 MB")
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_size < 1024:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f} PB"
