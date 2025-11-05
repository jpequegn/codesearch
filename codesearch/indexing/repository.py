"""Repository management for multi-repository indexing support.

Manages multiple indexed repositories with per-repository configuration,
namespace handling, and metadata tracking.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RepositoryMetadata:
    """Tracks metadata for an indexed repository."""

    def __init__(self, repo_id: str, repo_path: str, repo_name: str):
        """Initialize repository metadata.

        Args:
            repo_id: Unique repository identifier
            repo_path: Path to repository
            repo_name: Human-readable repository name
        """
        self.repo_id = repo_id
        self.repo_path = repo_path
        self.repo_name = repo_name
        self.indexed_at = datetime.now().isoformat()
        self.entity_count = 0
        self.file_count = 0
        self.namespace_prefix = self._generate_namespace(repo_id)

    def _generate_namespace(self, repo_id: str) -> str:
        """Generate unique namespace prefix for repository.

        Args:
            repo_id: Repository identifier

        Returns:
            Namespace prefix (e.g., 'repo_123')
        """
        return f"repo_{repo_id[:8]}"

    def to_dict(self) -> dict:
        """Convert to dictionary for persistence.

        Returns:
            Dictionary representation
        """
        return {
            "repo_id": self.repo_id,
            "repo_path": self.repo_path,
            "repo_name": self.repo_name,
            "indexed_at": self.indexed_at,
            "entity_count": self.entity_count,
            "file_count": self.file_count,
            "namespace_prefix": self.namespace_prefix,
        }

    @staticmethod
    def from_dict(data: dict) -> "RepositoryMetadata":
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            RepositoryMetadata instance
        """
        metadata = RepositoryMetadata(
            repo_id=data["repo_id"],
            repo_path=data["repo_path"],
            repo_name=data["repo_name"],
        )
        metadata.indexed_at = data.get("indexed_at", metadata.indexed_at)
        metadata.entity_count = data.get("entity_count", 0)
        metadata.file_count = data.get("file_count", 0)
        metadata.namespace_prefix = data.get(
            "namespace_prefix", metadata.namespace_prefix
        )
        return metadata


class RepositoryConfig:
    """Configuration for a single repository."""

    def __init__(self, repo_path: str, repo_name: Optional[str] = None):
        """Initialize repository configuration.

        Args:
            repo_path: Path to repository
            repo_name: Optional human-readable name (defaults to directory name)
        """
        self.repo_path = str(Path(repo_path).resolve())
        self.repo_name = repo_name or Path(repo_path).name
        self.repo_id = self._generate_repo_id()

        # Per-repo settings
        self.languages: List[str] = []  # Empty = index all
        self.exclude_patterns: List[str] = []
        self.index_tests = True
        self.min_complexity = 0

    def _generate_repo_id(self) -> str:
        """Generate unique repository ID from path.

        Returns:
            Repository ID (SHA256 hash of path)
        """
        path_hash = hashlib.sha256(self.repo_path.encode()).hexdigest()
        return path_hash[:16]

    def to_dict(self) -> dict:
        """Convert to dictionary for persistence.

        Returns:
            Dictionary representation
        """
        return {
            "repo_path": self.repo_path,
            "repo_name": self.repo_name,
            "repo_id": self.repo_id,
            "languages": self.languages,
            "exclude_patterns": self.exclude_patterns,
            "index_tests": self.index_tests,
            "min_complexity": self.min_complexity,
        }

    @staticmethod
    def from_dict(data: dict) -> "RepositoryConfig":
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            RepositoryConfig instance
        """
        config = RepositoryConfig(
            repo_path=data["repo_path"], repo_name=data.get("repo_name")
        )
        config.languages = data.get("languages", [])
        config.exclude_patterns = data.get("exclude_patterns", [])
        config.index_tests = data.get("index_tests", True)
        config.min_complexity = data.get("min_complexity", 0)
        return config


class RepositoryRegistry:
    """Manages multiple repository configurations and metadata."""

    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize repository registry.

        Args:
            registry_path: Path to registry file (default: ~/.codesearch/repositories.json)
        """
        self.registry_path = registry_path or (
            Path.home() / ".codesearch" / "repositories.json"
        )
        self.repositories: Dict[str, RepositoryConfig] = {}
        self.metadata: Dict[str, RepositoryMetadata] = {}
        self.load()

    def load(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                    for repo_id, repo_data in data.get("repositories", {}).items():
                        config = RepositoryConfig.from_dict(repo_data)
                        self.repositories[repo_id] = config

                    for repo_id, meta_data in data.get("metadata", {}).items():
                        metadata = RepositoryMetadata.from_dict(meta_data)
                        self.metadata[repo_id] = metadata

                logger.debug(f"Loaded registry with {len(self.repositories)} repos")
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                self.repositories = {}
                self.metadata = {}

    def save(self) -> None:
        """Save registry to disk."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "repositories": {
                    repo_id: config.to_dict()
                    for repo_id, config in self.repositories.items()
                },
                "metadata": {
                    repo_id: meta.to_dict()
                    for repo_id, meta in self.metadata.items()
                },
            }
            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved registry with {len(self.repositories)} repos")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register_repository(
        self, repo_path: str, repo_name: Optional[str] = None
    ) -> RepositoryConfig:
        """Register a new repository.

        Args:
            repo_path: Path to repository
            repo_name: Optional human-readable name

        Returns:
            RepositoryConfig for the registered repository

        Raises:
            ValueError: If repository already registered
        """
        config = RepositoryConfig(repo_path, repo_name)

        if config.repo_id in self.repositories:
            raise ValueError(f"Repository already registered: {config.repo_id}")

        self.repositories[config.repo_id] = config
        self.metadata[config.repo_id] = RepositoryMetadata(
            config.repo_id, config.repo_path, config.repo_name
        )
        self.save()

        logger.info(f"Registered repository: {config.repo_name} ({config.repo_id})")
        return config

    def unregister_repository(self, repo_id: str) -> bool:
        """Unregister a repository.

        Args:
            repo_id: Repository ID

        Returns:
            True if successful, False if not found
        """
        if repo_id not in self.repositories:
            return False

        del self.repositories[repo_id]
        if repo_id in self.metadata:
            del self.metadata[repo_id]

        self.save()
        logger.info(f"Unregistered repository: {repo_id}")
        return True

    def get_repository(self, repo_id: str) -> Optional[RepositoryConfig]:
        """Get repository configuration.

        Args:
            repo_id: Repository ID

        Returns:
            RepositoryConfig or None if not found
        """
        return self.repositories.get(repo_id)

    def get_metadata(self, repo_id: str) -> Optional[RepositoryMetadata]:
        """Get repository metadata.

        Args:
            repo_id: Repository ID

        Returns:
            RepositoryMetadata or None if not found
        """
        return self.metadata.get(repo_id)

    def list_repositories(self) -> List[RepositoryConfig]:
        """Get all registered repositories.

        Returns:
            List of RepositoryConfig objects
        """
        return list(self.repositories.values())

    def find_by_path(self, repo_path: str) -> Optional[RepositoryConfig]:
        """Find repository by path.

        Args:
            repo_path: Path to repository

        Returns:
            RepositoryConfig or None if not found
        """
        repo_path = str(Path(repo_path).resolve())
        for config in self.repositories.values():
            if config.repo_path == repo_path:
                return config
        return None

    def update_metadata(
        self, repo_id: str, entity_count: int = 0, file_count: int = 0
    ) -> None:
        """Update repository metadata after indexing.

        Args:
            repo_id: Repository ID
            entity_count: Number of indexed entities
            file_count: Number of indexed files
        """
        if repo_id in self.metadata:
            self.metadata[repo_id].entity_count += entity_count
            self.metadata[repo_id].file_count += file_count
            self.metadata[repo_id].indexed_at = datetime.now().isoformat()
            self.save()


class NamespaceManager:
    """Manages namespacing to avoid entity name collisions across repositories."""

    def __init__(self, registry: RepositoryRegistry):
        """Initialize namespace manager.

        Args:
            registry: RepositoryRegistry instance
        """
        self.registry = registry
        self.entity_namespaces: Dict[str, Set[str]] = {}  # repo_id -> entity names

    def add_entity(self, repo_id: str, entity_name: str) -> str:
        """Register an entity in a repository namespace.

        Args:
            repo_id: Repository ID
            entity_name: Entity name

        Returns:
            Fully qualified entity name (with namespace)
        """
        if repo_id not in self.entity_namespaces:
            self.entity_namespaces[repo_id] = set()

        self.entity_namespaces[repo_id].add(entity_name)
        return self._make_qualified(repo_id, entity_name)

    def get_qualified_name(self, repo_id: str, entity_name: str) -> str:
        """Get fully qualified name for an entity.

        Args:
            repo_id: Repository ID
            entity_name: Entity name

        Returns:
            Fully qualified entity name
        """
        return self._make_qualified(repo_id, entity_name)

    def resolve_entity(
        self, entity_name: str, search_repos: Optional[List[str]] = None
    ) -> List[Tuple[str, str]]:
        """Resolve entity name to (repo_id, entity_name) pairs.

        Handles both qualified and unqualified names.

        Args:
            entity_name: Entity name (with or without namespace)
            search_repos: Optional list of repo IDs to search (None = all)

        Returns:
            List of (repo_id, entity_name) tuples matching the entity
        """
        results = []
        repos_to_search = search_repos or list(self.entity_namespaces.keys())

        for repo_id in repos_to_search:
            if repo_id not in self.entity_namespaces:
                continue

            # Check for exact match
            if entity_name in self.entity_namespaces[repo_id]:
                results.append((repo_id, entity_name))

            # Check for qualified match
            qualified = self._make_qualified(repo_id, entity_name)
            if qualified.endswith(entity_name):
                results.append((repo_id, entity_name))

        return results

    def check_collision(self, repo_id: str, entity_name: str) -> List[str]:
        """Check if entity name exists in other repositories.

        Args:
            repo_id: Repository ID
            entity_name: Entity name

        Returns:
            List of repo IDs where entity name exists
        """
        collisions = []
        for other_repo_id, entities in self.entity_namespaces.items():
            if other_repo_id != repo_id and entity_name in entities:
                collisions.append(other_repo_id)
        return collisions

    def _make_qualified(self, repo_id: str, entity_name: str) -> str:
        """Make fully qualified name using repository namespace.

        Args:
            repo_id: Repository ID
            entity_name: Entity name

        Returns:
            Fully qualified name (e.g., repo_abc123:entity_name)
        """
        metadata = self.registry.get_metadata(repo_id)
        if not metadata:
            return entity_name
        return f"{metadata.namespace_prefix}:{entity_name}"
