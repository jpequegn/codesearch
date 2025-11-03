from typing import List, Optional
from codesearch.query.models import SearchResult
from codesearch.query.filters import MetadataFilter
from codesearch.query.exceptions import QueryError


class QueryEngine:
    """Orchestrates vector search with filtering and pagination."""

    def __init__(self, client):
        """Initialize QueryEngine with LanceDB client.

        Args:
            client: LanceDB client instance
        """
        self.client = client
        try:
            self.code_entities_table = client.get_table("code_entities")
        except Exception as e:
            raise QueryError(f"Failed to initialize QueryEngine: {str(e)}", e)

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
        # TODO: Embed query_text to vector using EmbeddingGenerator
        # For now, this is a placeholder
        raise NotImplementedError("Text search requires EmbeddingGenerator integration")

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

        return SearchResult(
            entity_id=raw_result['entity_id'],
            name=raw_result['name'],
            code_text=raw_result['code_text'],
            similarity_score=similarity,
            language=raw_result['language'],
            file_path=raw_result['file_path'],
            repository=raw_result['repository'],
            entity_type=raw_result['entity_type'],
            start_line=raw_result['start_line'],
            end_line=raw_result['end_line']
        )
