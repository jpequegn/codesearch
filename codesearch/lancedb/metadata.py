"""Search metadata operations."""

from typing import List, Optional
from codesearch.lancedb.models import SearchMetadata, ImplementationStatus
from codesearch.lancedb.client import LanceDBClient
import logging


logger = logging.getLogger(__name__)


class MetadataOperations:
    """Manage search_metadata table operations."""

    TABLE_NAME = "search_metadata"

    def __init__(self, client: LanceDBClient):
        self.client = client

    def _ensure_table_exists(self):
        """Create table if it doesn't exist."""
        # LanceDB requires data to create a table, so we defer creation
        # until the first insert
        pass

    def insert(self, metadata_list: List[SearchMetadata]) -> (int, List[str]):
        """Insert metadata with validation."""
        errors = []
        valid_items = []

        for metadata in metadata_list:
            validation_errors = self._validate_metadata(metadata)
            if validation_errors:
                errors.extend(validation_errors)
            else:
                valid_items.append(metadata.to_dict())

        if valid_items:
            try:
                self.client.insert(self.TABLE_NAME, valid_items)
                logger.info(f"Inserted {len(valid_items)} metadata records")
            except Exception as e:
                errors.append(f"Insert failed: {str(e)}")
                return 0, errors

        return len(valid_items), errors

    def get_by_id(self, metadata_id: str) -> Optional[SearchMetadata]:
        """Retrieve metadata by ID."""
        try:
            table = self.client.get_table(self.TABLE_NAME)
            results = table.search().where(f"metadata_id = '{metadata_id}'").limit(1).to_list()
            if results:
                return SearchMetadata(**results[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get metadata {metadata_id}: {e}")
            raise

    def search_by_status(self, status: ImplementationStatus, k: int = 100) -> List[SearchMetadata]:
        """Find metadata by implementation status."""
        try:
            table = self.client.get_table(self.TABLE_NAME)
            results = table.search().where(f"implementation_status = '{status.value}'").limit(k).to_list()
            return [SearchMetadata(**r) for r in results]
        except Exception as e:
            logger.error(f"Failed to search by status: {e}")
            raise

    def search_by_feature_area(self, feature_area: str, k: int = 100) -> List[SearchMetadata]:
        """Find metadata by feature area."""
        try:
            table = self.client.get_table(self.TABLE_NAME)
            results = table.search().where(f"feature_area = '{feature_area}'").limit(k).to_list()
            return [SearchMetadata(**r) for r in results]
        except Exception as e:
            logger.error(f"Failed to search by feature area: {e}")
            raise

    def count(self) -> int:
        """Get total metadata count."""
        try:
            table = self.client.get_table(self.TABLE_NAME)
            return len(table.to_pandas())
        except Exception as e:
            logger.error(f"Count failed: {e}")
            raise

    @staticmethod
    def _validate_metadata(metadata: SearchMetadata) -> List[str]:
        """Validate metadata before insertion."""
        errors = []

        if not metadata.metadata_id:
            errors.append("Missing metadata_id")
        if not metadata.entity_id:
            errors.append("Missing entity_id")
        if not metadata.repository:
            errors.append("Missing repository")

        return errors
