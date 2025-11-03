"""Data models for query results."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    """Represents a single search result from the code search engine.

    Attributes:
        entity_id: Unique identifier for the code entity
        name: Name of the function/class/method
        code_text: Source code of the entity
        similarity_score: Cosine similarity score (0.0-1.0)
        language: Programming language
        file_path: Path to the source file
        repository: Repository name or path
        entity_type: Type of entity (function, class, method, etc.)
        start_line: Starting line number in source file
        end_line: Ending line number in source file
    """

    entity_id: str
    name: str
    code_text: str
    similarity_score: float
    language: str
    file_path: str
    repository: str
    entity_type: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "code_text": self.code_text,
            "similarity_score": self.similarity_score,
            "language": self.language,
            "file_path": self.file_path,
            "repository": self.repository,
            "entity_type": self.entity_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }
