"""Repository scanner for finding and filtering source files."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from codesearch.models import FileMetadata, RepositoryScanner


class RepositoryScannerImpl:
    """Implementation of repository scanning with file discovery and filtering."""

    def __init__(self, config: Optional[RepositoryScanner] = None):
        """
        Initialize the repository scanner.

        Args:
            config: RepositoryScanner configuration (uses defaults if None)
        """
        self.config = config or RepositoryScanner()
        self.scanned_files: Dict[str, List[FileMetadata]] = {}

    def scan_repository(
        self, repo_path: str, repo_name: Optional[str] = None
    ) -> List[FileMetadata]:
        """
        Scan a repository and return discovered files.

        Args:
            repo_path: Path to repository root
            repo_name: Optional name for the repository

        Returns:
            List of FileMetadata for discovered files
        """
        if not os.path.isdir(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")

        repo_name = repo_name or os.path.basename(repo_path)
        self.config.add_repository(repo_name, repo_path)

        files = self._walk_directory(repo_path)
        self.scanned_files[repo_name] = files

        return files

    def scan_multiple_repositories(
        self, repo_paths: Dict[str, str]
    ) -> Dict[str, List[FileMetadata]]:
        """
        Scan multiple repositories.

        Args:
            repo_paths: Dict mapping repo names to paths

        Returns:
            Dict mapping repo names to lists of FileMetadata
        """
        results = {}
        for repo_name, repo_path in repo_paths.items():
            results[repo_name] = self.scan_repository(repo_path, repo_name)
        return results

    def _walk_directory(self, root_path: str) -> List[FileMetadata]:
        """
        Walk directory tree and collect files.

        Args:
            root_path: Root directory to walk from

        Returns:
            List of FileMetadata for all discovered files
        """
        files = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Filter directories - remove ignored ones in-place to prevent descent
            # This modifies dirnames in-place, preventing os.walk from descending
            dirnames[:] = [d for d in dirnames if self.config.should_scan_directory(d)]

            # Process files in this directory
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                metadata = self._get_file_metadata(file_path, root_path)

                if metadata and not metadata.is_ignored:
                    files.append(metadata)

        return files

    def _get_file_metadata(
        self, file_path: str, root_path: str
    ) -> Optional[FileMetadata]:
        """
        Get metadata for a single file.

        Args:
            file_path: Full path to file
            root_path: Root repository path (for relative paths)

        Returns:
            FileMetadata or None if file should be ignored
        """
        try:
            # Detect language
            language = self.config.get_language_from_extension(file_path)
            if language is None:
                return None  # Unsupported file type

            # Check if file should be scanned
            if not self.config.should_scan_file(file_path, language):
                return FileMetadata(
                    file_path=file_path,
                    language=language,
                    size_bytes=0,
                    modified_time=datetime.now(),
                    is_ignored=True,
                    ignore_reason="File matches ignore pattern",
                )

            # Get file stats
            stat = os.stat(file_path)
            size_bytes = stat.st_size
            modified_time = datetime.fromtimestamp(stat.st_mtime)

            return FileMetadata(
                file_path=file_path,
                language=language,
                size_bytes=size_bytes,
                modified_time=modified_time,
                is_ignored=False,
            )

        except (OSError, IOError) as e:
            # Return None for files that can't be accessed
            return None

    def get_files_by_language(self) -> Dict[str, List[FileMetadata]]:
        """
        Get all discovered files grouped by language.

        Returns:
            Dict mapping language names to lists of FileMetadata
        """
        files_by_lang: Dict[str, List[FileMetadata]] = {}

        for repo_files in self.scanned_files.values():
            for file_metadata in repo_files:
                if file_metadata.language not in files_by_lang:
                    files_by_lang[file_metadata.language] = []
                files_by_lang[file_metadata.language].append(file_metadata)

        return files_by_lang

    def get_python_files(self) -> List[FileMetadata]:
        """
        Get all Python files discovered.

        Returns:
            List of FileMetadata for Python files
        """
        python_files = []
        for repo_files in self.scanned_files.values():
            for file_metadata in repo_files:
                if file_metadata.language == "python":
                    python_files.append(file_metadata)
        return python_files

    def get_statistics(self) -> Dict[str, object]:
        """
        Get scanning statistics.

        Returns:
            Dict with scanning stats (total files, by language, by repository)
        """
        stats = {
            "total_files": 0,
            "by_repository": {},
            "by_language": {},
            "total_size_bytes": 0,
        }

        # Count by repository
        for repo_name, files in self.scanned_files.items():
            stats["by_repository"][repo_name] = len(files)
            stats["total_files"] += len(files)
            stats["total_size_bytes"] += sum(f.size_bytes for f in files)

        # Count by language
        files_by_lang = self.get_files_by_language()
        for lang, files in files_by_lang.items():
            stats["by_language"][lang] = len(files)

        return stats

    def get_repository_path(self, repo_name: str) -> Optional[str]:
        """
        Get the file system path for a repository.

        Args:
            repo_name: Repository name/identifier

        Returns:
            File system path or None if not found
        """
        return self.config.repositories.get(repo_name)

    def clear_scanned_files(self) -> None:
        """Clear all scanned file records."""
        self.scanned_files.clear()

    def add_ignored_directory(self, dir_name: str) -> None:
        """
        Add a directory to the ignore list.

        Args:
            dir_name: Directory name to ignore
        """
        self.config.ignored_directories.add(dir_name)

    def remove_ignored_directory(self, dir_name: str) -> None:
        """
        Remove a directory from the ignore list.

        Args:
            dir_name: Directory name to stop ignoring
        """
        self.config.ignored_directories.discard(dir_name)

    def add_supported_language(self, language: str) -> None:
        """
        Add a language to the supported list.

        Args:
            language: Language name (e.g., 'python', 'javascript')
        """
        self.config.supported_languages.add(language)

    def remove_supported_language(self, language: str) -> None:
        """
        Remove a language from the supported list.

        Args:
            language: Language name to stop supporting
        """
        self.config.supported_languages.discard(language)
