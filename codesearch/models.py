"""Data models for codesearch."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Class:
    """Represents a class extracted from source code."""

    name: str
    file_path: str
    language: str = "python"

    # Code content
    source_code: str = ""
    docstring: Optional[str] = None

    # Metadata
    line_number: int = 0
    end_line: int = 0

    # Base classes
    bases: List[str] = field(default_factory=list)

    # Embedding
    embedding: Optional[List[float]] = None

    def __hash__(self):
        """Make Class hashable for use in sets."""
        return hash((self.name, self.file_path, self.line_number))

    def __eq__(self, other):
        """Compare classes by identity."""
        if not isinstance(other, Class):
            return False
        return (
            self.name == other.name
            and self.file_path == other.file_path
            and self.line_number == other.line_number
        )


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

    # Class context (if method)
    class_name: Optional[str] = None
    is_method: bool = False
    is_async: bool = False

    # Nesting information
    parent_function: Optional[str] = None  # For nested functions
    depth: int = 0  # 0 for top-level, 1+ for nested

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

    def fully_qualified_name(self) -> str:
        """Return fully qualified name for nested functions/methods."""
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        elif self.parent_function:
            return f"{self.parent_function}.{self.name}"
        return self.name
