"""Code relationship operations for call graphs."""

from typing import List, Optional
from codesearch.lancedb.models import CodeRelationship, RelationshipType
from codesearch.lancedb.client import LanceDBClient
import logging


logger = logging.getLogger(__name__)


class RelationshipOperations:
    """Manage code_relationships table operations."""

    TABLE_NAME = "code_relationships"

    def __init__(self, client: LanceDBClient):
        self.client = client

    def _ensure_table_exists(self):
        """Create table if it doesn't exist."""
        # LanceDB requires data to create a table, so we defer creation
        # until the first insert
        pass

    def insert(self, relationships: List[CodeRelationship]) -> (int, List[str]):
        """Insert relationships with validation."""
        errors = []
        valid_rels = []

        for rel in relationships:
            validation_errors = self._validate_relationship(rel)
            if validation_errors:
                errors.extend(validation_errors)
            else:
                valid_rels.append(rel.to_dict())

        if valid_rels:
            try:
                self.client.insert(self.TABLE_NAME, valid_rels)
                logger.info(f"Inserted {len(valid_rels)} relationships")
            except Exception as e:
                errors.append(f"Insert failed: {str(e)}")
                return 0, errors

        return len(valid_rels), errors

    def get_callees(self, caller_id: str) -> List[CodeRelationship]:
        """Get all functions called by a function."""
        try:
            table = self.client.get_table(self.TABLE_NAME)
            results = table.search().where(f"caller_id = '{caller_id}'").to_list()
            return [CodeRelationship(**r) for r in results]
        except Exception as e:
            logger.error(f"Failed to get callees: {e}")
            raise

    def get_callers(self, callee_id: str) -> List[CodeRelationship]:
        """Get all functions that call a function."""
        try:
            table = self.client.get_table(self.TABLE_NAME)
            results = table.search().where(f"callee_id = '{callee_id}'").to_list()
            return [CodeRelationship(**r) for r in results]
        except Exception as e:
            logger.error(f"Failed to get callers: {e}")
            raise

    def count(self) -> int:
        """Get total relationship count."""
        try:
            table = self.client.get_table(self.TABLE_NAME)
            return len(table.to_pandas())
        except Exception as e:
            logger.error(f"Count failed: {e}")
            raise

    @staticmethod
    def _validate_relationship(rel: CodeRelationship) -> List[str]:
        """Validate relationship before insertion."""
        errors = []

        if not rel.relationship_id:
            errors.append("Missing relationship_id")
        if not rel.caller_id:
            errors.append("Missing caller_id")
        if not rel.callee_id:
            errors.append("Missing callee_id")
        if rel.caller_id == rel.callee_id:
            errors.append("Self-referential relationship")

        return errors
