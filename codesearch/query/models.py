from dataclasses import dataclass


@dataclass
class SearchResult:
    """Represents a single search result from code search."""

    entity_id: str
    name: str
    code_text: str
    similarity_score: float          # [0.0, 1.0], higher is better
    language: str
    file_path: str
    repository: str
    entity_type: str                # function, class, method, variable, module
    start_line: int
    end_line: int

    def __str__(self) -> str:
        """Human-readable representation for CLI output."""
        return (f"{self.name} ({self.language}) - {self.repository}:{self.file_path} "
                f"[similarity: {self.similarity_score:.3f}]")
