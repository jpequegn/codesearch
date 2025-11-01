"""Data models for codesearch."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Function:
    """Represents a function extracted from source code."""

    name: str
    file_path: str
    language: str = "python"

    # Code content
    source_code: str = ""
    docstring: Optional[str] = None

    # Metadata
    line_number: int = 0
    end_line: int = 0
    signature: str = ""

    # Call graph
    calls_to: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)

    # Embedding
    embedding: Optional[List[float]] = None

    def __hash__(self):
        """Make Function hashable for use in sets."""
        return hash((self.name, self.file_path, self.line_number))

    def __eq__(self, other):
        """Compare functions by identity."""
        if not isinstance(other, Function):
            return False
        return (
            self.name == other.name
            and self.file_path == other.file_path
            and self.line_number == other.line_number
        )
