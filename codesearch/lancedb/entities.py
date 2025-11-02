"""Code entity operations."""

from typing import List, Optional
from codesearch.lancedb.models import CodeEntity, EntityType
from codesearch.lancedb.client import LanceDBClient
import logging


logger = logging.getLogger(__name__)


class EntityOperations:
    """Manage code_entities table operations."""

    TABLE_NAME = "code_entities"

    def __init__(self, client: LanceDBClient):
        self.client = client

    def _ensure_table_exists(self):
        """Create table if it doesn't exist."""
        # LanceDB requires data to create a table, so we defer creation
        # until the first insert
        pass

    def insert(self, entities: List[CodeEntity]) -> (int, List[str]):
        """Insert entities with validation."""
        errors = []
        valid_entities = []

        for entity in entities:
            validation_errors = self._validate_entity(entity)
            if validation_errors:
                errors.extend(validation_errors)
            else:
                valid_entities.append(entity.to_dict())

        if valid_entities:
            try:
                self.client.insert(self.TABLE_NAME, valid_entities)
                logger.info(f"Inserted {len(valid_entities)} entities")
            except Exception as e:
                errors.append(f"Insert failed: {str(e)}")
                return 0, errors

        return len(valid_entities), errors

    def get_by_id(self, entity_id: str) -> Optional[CodeEntity]:
        """Retrieve entity by ID."""
        try:
            table = self.client.get_table(self.TABLE_NAME)
            results = table.search().where(f"entity_id = '{entity_id}'").limit(1).to_list()
            if results:
                return CodeEntity(**results[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
            raise

    def search_semantic(
        self, query_vector: List[float], k: int = 10, distance_threshold: float = 0.5
    ) -> List[CodeEntity]:
        """Semantic search using vector similarity."""
        try:
            table = self.client.get_table(self.TABLE_NAME)
            results = (
                table.search(query_vector)
                .limit(k)
                .to_list()
            )
            return [CodeEntity(**r) for r in results if r.get("_distance", 1.0) < distance_threshold]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise

    def search_filtered(
        self,
        repository: Optional[str] = None,
        language: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        k: int = 100,
    ) -> List[CodeEntity]:
        """Search with filters."""
        try:
            table = self.client.get_table(self.TABLE_NAME)
            query = table.search()

            if repository:
                query = query.where(f"repository = '{repository}'")
            if language:
                query = query.where(f"language = '{language}'")
            if entity_type:
                query = query.where(f"entity_type = '{entity_type.value}'")

            results = query.limit(k).to_list()
            return [CodeEntity(**r) for r in results]
        except Exception as e:
            logger.error(f"Filtered search failed: {e}")
            raise

    def count(self) -> int:
        """Get total entity count."""
        try:
            table = self.client.get_table(self.TABLE_NAME)
            return len(table.to_pandas())
        except Exception as e:
            logger.error(f"Count failed: {e}")
            raise

    @staticmethod
    def _validate_entity(entity: CodeEntity) -> List[str]:
        """Validate entity before insertion."""
        errors = []

        if not entity.entity_id:
            errors.append("Missing entity_id")
        if not entity.code_vector:
            errors.append("Missing code_vector")

        if entity.code_vector:
            if len(entity.code_vector) != 768:
                errors.append(f"Vector dimension mismatch: {len(entity.code_vector)} != 768")
            if not all(-2.0 <= v <= 2.0 for v in entity.code_vector):
                errors.append("Vector values out of range [-2.0, 2.0]")

        return errors
