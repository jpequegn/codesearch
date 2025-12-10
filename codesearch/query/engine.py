from typing import List, Optional

from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.query.exceptions import QueryError
from codesearch.query.filters import MetadataFilter
from codesearch.query.models import SearchResult


class QueryEngine:
    """Orchestrates vector search with filtering and pagination."""

    def __init__(self, client, embedder: Optional[EmbeddingGenerator] = None):
        """Initialize QueryEngine with LanceDB client.

        Args:
            client: LanceDB client instance
            embedder: Optional EmbeddingGenerator for text search.
                      If not provided, one will be created on first text search.
        """
        self.client = client
        self._embedder = embedder
        try:
            self.code_entities_table = client.open_table("code_entities")
        except Exception as e:
            raise QueryError(f"Failed to initialize QueryEngine: {str(e)}", e)

    @property
    def embedder(self) -> EmbeddingGenerator:
        """Lazy-load embedding generator on first use."""
        if self._embedder is None:
            self._embedder = EmbeddingGenerator()
        return self._embedder

    def search_vector(
        self,
        query_vector: List[float],
        filters: Optional[List[MetadataFilter]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[SearchResult]:
        """Search by pre-computed vector with optional filtering.

        Args:
            query_vector: 768-dimensional query embedding
            filters: List of MetadataFilter objects (optional)
            limit: Number of results to return (default 10)
            offset: Number of results to skip (default 0)

        Returns:
            List[SearchResult] sorted by similarity (best first)
        """
        try:
            # Fetch more results to account for filtering dropout
            fetch_limit = limit * 2 if filters else limit

            # Vector search via LanceDB
            raw_results = (
                self.code_entities_table
                .search(query_vector)
                .limit(fetch_limit)
                .to_list()
            )

            # Convert to SearchResult objects
            results = [self._raw_to_search_result(r) for r in raw_results]

            # Apply metadata filters
            if filters:
                results = self._apply_filters(results, filters)

            # Apply pagination
            paginated = results[offset:offset + limit]

            return paginated
        except QueryError:
            raise
        except Exception as e:
            raise QueryError(f"Vector search failed: {str(e)}", e)

    def search_text(
        self,
        query_text: str,
        filters: Optional[List[MetadataFilter]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[SearchResult]:
        """Search by text (embedded automatically).

        Args:
            query_text: Natural language query
            filters: List of MetadataFilter objects (optional)
            limit: Number of results to return (default 10)
            offset: Number of results to skip (default 0)

        Returns:
            List[SearchResult] sorted by similarity
        """
        try:
            # Embed query text to vector
            query_vector = self.embedder.embed_code(query_text)

            # Delegate to vector search
            return self.search_vector(
                query_vector=query_vector,
                filters=filters,
                limit=limit,
                offset=offset
            )
        except QueryError:
            raise
        except Exception as e:
            raise QueryError(f"Text search failed: {str(e)}", e) from e

    def _apply_filters(
        self,
        results: List[SearchResult],
        filters: List[MetadataFilter]
    ) -> List[SearchResult]:
        """Apply metadata filters using AND logic between filter types.

        Args:
            results: List of SearchResult to filter
            filters: List of MetadataFilter objects

        Returns:
            Filtered results that match ALL filters
        """
        filtered = results
        for filter_obj in filters:
            filtered = [r for r in filtered if filter_obj.matches(r)]
        return filtered

    def _raw_to_search_result(self, raw_result: dict) -> SearchResult:
        """Convert raw LanceDB result to SearchResult.

        Args:
            raw_result: Dictionary from LanceDB search

        Returns:
            SearchResult object
        """
        # LanceDB returns _distance (lower is better)
        # Convert to similarity_score (higher is better: 1 - distance)
        distance = raw_result.get('_distance', 0.0)
        similarity = 1.0 - min(distance, 1.0)

        # Handle both old schema (start_line/end_line) and new schema (line_count)
        start_line = raw_result.get('start_line', 0)
        end_line = raw_result.get('end_line', 0)

        # If we only have line_count, estimate end_line
        if start_line == 0 and end_line == 0:
            line_count = raw_result.get('line_count', 0)
            # Without start_line info, we can't determine exact position
            start_line = 1
            end_line = line_count if line_count > 0 else 1

        return SearchResult(
            entity_id=raw_result['entity_id'],
            name=raw_result['name'],
            code_text=raw_result['code_text'],
            similarity_score=similarity,
            language=raw_result['language'],
            file_path=raw_result['file_path'],
            repository=raw_result['repository'],
            entity_type=raw_result['entity_type'],
            start_line=start_line,
            end_line=end_line
        )
