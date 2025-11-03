from abc import ABC, abstractmethod
from typing import List


class MetadataFilter(ABC):
    """Abstract base class for metadata filters."""

    @abstractmethod
    def matches(self, result) -> bool:
        """Check if result matches this filter."""
        pass


class LanguageFilter(MetadataFilter):
    """Filter by programming language (OR logic)."""

    def __init__(self, languages: List[str]):
        """Initialize with list of allowed languages."""
        self.languages = languages

    def matches(self, result) -> bool:
        """Check if result language is in allowed list."""
        return result.language in self.languages


class FilePathFilter(MetadataFilter):
    """Filter by file path substring (OR logic)."""

    def __init__(self, path_patterns: List[str]):
        """Initialize with list of path patterns."""
        self.patterns = path_patterns

    def matches(self, result) -> bool:
        """Check if result path contains any pattern."""
        return any(pattern in result.file_path for pattern in self.patterns)


class EntityTypeFilter(MetadataFilter):
    """Filter by entity type (function, class, etc.)."""

    def __init__(self, entity_types: List[str]):
        """Initialize with list of allowed entity types."""
        self.types = entity_types

    def matches(self, result) -> bool:
        """Check if result entity_type is in allowed list."""
        return result.entity_type in self.types


class RepositoryFilter(MetadataFilter):
    """Filter by repository name."""

    def __init__(self, repositories: List[str]):
        """Initialize with list of allowed repositories."""
        self.repos = repositories

    def matches(self, result) -> bool:
        """Check if result repository is in allowed list."""
        return result.repository in self.repos
