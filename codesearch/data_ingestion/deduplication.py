from typing import Set, Optional
from codesearch.models import CodeEntity


class DeduplicationCache:
    """Fast duplicate detection via source_hash lookup."""

    def __init__(self, client):
        """Initialize cache by loading existing source_hashes from database.

        Args:
            client: LanceDB client instance
        """
        self.client = client
        self.hashes: Set[str] = set()
        self._load_from_database()

    def _load_from_database(self) -> None:
        """Load all existing source_hashes from code_entities table.

        This ensures deduplication cache is consistent with database state.
        Called during initialization to populate hashes set.
        """
        # TODO: Implement database loading
        # For now, start with empty set
        self.hashes = set()

    def is_duplicate(self, entity: CodeEntity) -> bool:
        """Check if entity's source_hash is already known.

        Args:
            entity: CodeEntity to check

        Returns:
            True if source_hash already exists, False otherwise
        """
        return entity.source_hash in self.hashes

    def add(self, entity: CodeEntity) -> None:
        """Add entity's source_hash to cache after successful insert.

        Args:
            entity: CodeEntity that was successfully inserted
        """
        self.hashes.add(entity.source_hash)

    def refresh(self) -> None:
        """Reload hashes from database for multi-process safety.

        Call this periodically or between batch operations to ensure
        cache stays consistent with concurrent database updates.
        """
        self._load_from_database()

    def detect_update(self, entity: CodeEntity, existing: CodeEntity) -> bool:
        """Detect if entity is an update to existing record.

        An update is detected when:
        - source_hash matches (code is identical)
        - But metadata differs (repository, file_path, or updated_at)

        Args:
            entity: New incoming entity
            existing: Entity already in database

        Returns:
            True if this is an update (same code, different metadata)
        """
        if entity.source_hash != existing.source_hash:
            return False

        # Check if metadata differs
        return (entity.repository != existing.repository or
                entity.file_path != existing.file_path)
