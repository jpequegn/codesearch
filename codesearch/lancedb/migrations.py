"""Schema migrations and consistency validation."""

from typing import Callable, List, Dict, Any
import logging


logger = logging.getLogger(__name__)


class MigrationRegistry:
    """Registry of schema migrations."""

    def __init__(self):
        self._migrations: Dict[str, Callable] = {}

    def register(self, version: str, migration_func: Callable):
        """Register a migration function."""
        self._migrations[version] = migration_func
        logger.info(f"Registered migration: {version}")

    def execute(self, version: str, db):
        """Execute migration for specific version."""
        if version not in self._migrations:
            raise ValueError(f"Migration {version} not found")

        try:
            logger.info(f"Executing migration: {version}")
            self._migrations[version](db)
            logger.info(f"Migration successful: {version}")
        except Exception as e:
            logger.error(f"Migration failed: {version} - {e}")
            raise


class ConsistencyValidator:
    """Validate and verify consistency across tables."""

    def __init__(self, client):
        self.client = client

    def validate_all(self) -> Dict[str, Any]:
        """Validate all consistency invariants."""
        from datetime import datetime, timezone

        report = {
            "valid": True,
            "entity_issues": [],
            "relationship_issues": [],
            "metadata_issues": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            entity_issues = self._check_entity_integrity()
            report["entity_issues"] = entity_issues
            if entity_issues:
                report["valid"] = False
        except Exception as e:
            report["entity_issues"].append(f"Check failed: {e}")
            report["valid"] = False

        try:
            rel_issues = self._check_relationship_integrity()
            report["relationship_issues"] = rel_issues
            if rel_issues:
                report["valid"] = False
        except Exception as e:
            report["relationship_issues"].append(f"Check failed: {e}")
            report["valid"] = False

        try:
            meta_issues = self._check_metadata_integrity()
            report["metadata_issues"] = meta_issues
            if meta_issues:
                report["valid"] = False
        except Exception as e:
            report["metadata_issues"].append(f"Check failed: {e}")
            report["valid"] = False

        return report

    def _check_entity_integrity(self) -> List[str]:
        """Verify entity table integrity."""
        issues = []
        try:
            if self.client.table_exists("code_entities"):
                table = self.client.get_table("code_entities")
                df = table.to_pandas()

                # Check for duplicates
                if len(df) != len(df.drop_duplicates("entity_id")):
                    issues.append("Duplicate entity_id found")

                # Check for missing vectors
                if "code_vector" in df.columns:
                    missing = df["code_vector"].isna().sum()
                    if missing > 0:
                        issues.append(f"Missing vectors: {missing}")
        except Exception as e:
            issues.append(f"Failed to check entities: {e}")

        return issues

    def _check_relationship_integrity(self) -> List[str]:
        """Verify relationship table integrity."""
        issues = []
        try:
            if self.client.table_exists("code_relationships"):
                rels = self.client.get_table("code_relationships").to_pandas()

                # Check for self-references
                if len(rels) > 0:
                    self_refs = (rels["caller_id"] == rels["callee_id"]).sum()
                    if self_refs > 0:
                        issues.append(f"Self-referential relationships: {self_refs}")
        except Exception as e:
            issues.append(f"Failed to check relationships: {e}")

        return issues

    def _check_metadata_integrity(self) -> List[str]:
        """Verify metadata table integrity."""
        issues = []
        try:
            if self.client.table_exists("search_metadata"):
                meta = self.client.get_table("search_metadata").to_pandas()

                # Check for required fields
                if len(meta) > 0:
                    for col in ["metadata_id", "entity_id", "repository"]:
                        if col in meta.columns:
                            missing = meta[col].isna().sum()
                            if missing > 0:
                                issues.append(f"Missing {col}: {missing}")
        except Exception as e:
            issues.append(f"Failed to check metadata: {e}")

        return issues
