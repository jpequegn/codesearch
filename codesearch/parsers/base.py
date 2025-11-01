"""Base parser abstraction for language-specific implementations."""

from abc import ABC, abstractmethod
from typing import List

from codesearch.models import Function


class BaseParser(ABC):
    """Abstract base class for code parsers."""

    @abstractmethod
    def parse_file(self, file_path: str) -> List[Function]:
        """
        Parse a single file and extract functions.

        Args:
            file_path: Path to the source file

        Returns:
            List of Function objects extracted from the file
        """
        pass

    @abstractmethod
    def get_language(self) -> str:
        """
        Return the language this parser handles.

        Returns:
            Language name (e.g., 'python', 'typescript')
        """
        pass
