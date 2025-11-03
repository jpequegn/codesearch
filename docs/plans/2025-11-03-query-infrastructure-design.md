# Query Infrastructure Design - Issue #11

**Date**: 2025-11-03
**Component**: 3.3 Query Infrastructure
**Status**: Design Complete
**Approach**: Simple Query Layer with Direct LanceDB Integration

## Executive Summary

The Query Infrastructure provides a straightforward interface for semantic code search with vector similarity ranking and flexible metadata filtering. Built on LanceDB's vector search capabilities, it delivers:

- **Vector similarity search**: Find semantically similar code using 768-dim embeddings
- **Metadata filtering**: Filter by language, file path, entity type, repository (OR logic)
- **Offset pagination**: Simple page-based result navigation
- **Balanced performance**: 100-500ms typical query latency
- **Extensibility foundation**: Ready for future caching and ranking optimizations

## Architecture Overview

### Core Design: Simple Query Layer

The architecture leverages LanceDB's efficient vector search with Python-level metadata filtering:

```
User Query
    ↓
[Text → Vector Embedding]
    ↓
[LanceDB Vector Search]
    ↓
[Apply Metadata Filters]
    ↓
[Pagination (offset/limit)]
    ↓
[Return SearchResult List]
```

**Why Simple?** No query caching, no ranking customization, no query planning. Direct LanceDB operations with Python filtering gives us:
- Fast MVP reaching 100-500ms target
- Clear, maintainable code
- Strong foundation for v2 optimizations (caching, reranking)
- Sufficient for typical code search workloads

**Key principle**: Vector similarity is the primary ranking signal. Metadata filters act as inclusion/exclusion gates, not ranking factors.

## Component Details

### 1. QueryEngine

**Responsibility**: Orchestrate vector search with filtering and pagination

**Interface**:
```python
class QueryEngine:
    def __init__(self, client: LanceDBClient):
        """Initialize with LanceDB client."""
        self.client = client
        self.code_entities_table = client.get_table("code_entities")
        self.embedding_generator = EmbeddingGenerator()  # From embeddings module

    def search_vector(
        self,
        query_vector: List[float],
        filters: List[MetadataFilter] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[SearchResult]:
        """Search by pre-computed vector with optional filtering."""

    def search_text(
        self,
        query_text: str,
        filters: List[MetadataFilter] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[SearchResult]:
        """Search by text (embedded automatically)."""
```

**Search Flow**:
1. Accept query (text or vector)
2. If text: embed to 768-dim vector
3. Vector search: `table.search(query_vector).limit(limit * 2).to_list()`
   - Fetch 2x limit to account for filter dropout
4. Apply metadata filters (OR logic)
5. Slice by offset and limit
6. Convert to SearchResult objects
7. Return sorted by similarity_score (descending)

**Performance characteristics**:
- Vector search: <50ms (LanceDB indexed search)
- Filter application: <10ms (Python in-memory filtering)
- Total: 50-100ms typical, <500ms worst case
- Scales to millions of entities (LanceDB manages indexing)

### 2. MetadataFilter (Abstract Base)

**Responsibility**: Represent and apply metadata filtering constraints

**Supported Filters**:
```python
class LanguageFilter(MetadataFilter):
    """Filter by programming language (OR logic)."""
    def __init__(self, languages: List[str]):
        self.languages = languages
    def matches(self, result: CodeEntity) -> bool:
        return result.language in self.languages

class FilePathFilter(MetadataFilter):
    """Filter by file path substring (OR logic)."""
    def __init__(self, path_patterns: List[str]):
        self.patterns = path_patterns
    def matches(self, result: CodeEntity) -> bool:
        return any(pattern in result.file_path for pattern in self.patterns)

class EntityTypeFilter(MetadataFilter):
    """Filter by entity type: function, class, method, etc."""
    def __init__(self, entity_types: List[str]):
        self.types = entity_types
    def matches(self, result: CodeEntity) -> bool:
        return result.entity_type in self.types

class RepositoryFilter(MetadataFilter):
    """Filter by repository name."""
    def __init__(self, repositories: List[str]):
        self.repos = repositories
    def matches(self, result: CodeEntity) -> bool:
        return result.repository in self.repos
```

**Filter Logic**:
- Within a filter type: OR logic (language in [python, javascript] means python OR javascript)
- Between filter types: AND logic (LanguageFilter([python]) AND FilePathFilter(['src/']))
- This provides inclusive search (broader results) within types, strict matching across types

### 3. SearchResult

**Responsibility**: Represent a single search result

**Structure**:
```python
@dataclass
class SearchResult:
    entity_id: str
    name: str
    code_text: str
    similarity_score: float          # [0.0, 1.0], higher is better
    language: str
    file_path: str
    repository: str
    entity_type: str               # function, class, method, variable, module
    start_line: int
    end_line: int
```

**Methods**:
- `__str__()`: Human-readable representation for CLI output
- Properties for convenient access to nested data

### 4. Exceptions

**QueryError**: Raised for database/LanceDB errors
- Missing table, connection failure, malformed query
- Includes error context and suggestions

**FilterError**: Raised for invalid filter specifications
- Unknown filter type, invalid language, malformed pattern
- Clear error message guiding user to correct usage

## Data Flow: Step by Step

### Basic Vector Search

```python
engine = QueryEngine(client)

# Results already sorted by LanceDB similarity
results = engine.search_vector(
    query_vector=[0.1, 0.2, ...],  # 768 dimensions
    limit=10
)
# Returns: [SearchResult, SearchResult, ...] sorted by similarity_score desc
```

### Text Search

```python
results = engine.search_text(
    query_text="parse JSON function",
    limit=10
)

# Internally:
# 1. embedding_generator.embed("parse JSON function") → 768-dim vector
# 2. search_vector(vector, limit=10)
# 3. Return results
```

### Filtered Search (AND/OR Logic)

```python
results = engine.search_text(
    query_text="utility function",
    filters=[
        LanguageFilter(['python', 'javascript']),  # python OR javascript
        FilePathFilter(['src/', 'lib/'])           # src/ OR lib/
    ],
    limit=10
)

# Filter logic:
# - Keep results where (language=python OR language=javascript)
#           AND (file_path contains 'src/' OR file_path contains 'lib/')
```

### Pagination

```python
# Page 1 (results 0-9)
page1 = engine.search_text(query, limit=10, offset=0)

# Page 2 (results 10-19)
page2 = engine.search_text(query, limit=10, offset=10)

# Each call re-runs vector search and filtering (acceptable for typical page sizes)
```

## Performance Characteristics

### Latency Targets (Balanced: 100-500ms)

**Typical case** (search + filter + paginate):
- Vector search (LanceDB): 30-50ms
- Filter application (Python): 5-15ms
- Pagination (slice): <1ms
- **Total**: 35-65ms ✅

**Worst case** (search large result set, complex filters):
- Vector search: 50ms
- Filter application: 100-200ms
- Pagination: <1ms
- **Total**: 150-250ms ✅

**Why efficient?**
- LanceDB uses HNSW indexing for O(log n) search
- Filtering is in-memory O(n) after search
- Pagination is O(limit) slice operation

### Throughput

- Single query: <500ms
- Queries per second: >2 QPS per server process
- Horizontal scaling: Run multiple server instances behind load balancer

### Memory Usage

- Query engine: <10MB (client connection, caches)
- Search result: <1KB per result
- Typical result set: 100 results = 100KB
- No memory leaks (results garbage collected after response)

## Error Handling

### Connection Errors

```python
try:
    results = engine.search_text(query)
except QueryError as e:
    # Handle: "Cannot connect to LanceDB: ..."
    # Action: Retry with backoff, check database status
```

### Table Missing

```python
try:
    engine = QueryEngine(client)  # Validates tables on init
except QueryError as e:
    # Handle: "Table 'code_entities' not found"
    # Action: Run data ingestion pipeline first
```

### Invalid Filters

```python
try:
    results = engine.search_text(
        query,
        filters=[LanguageFilter(['cobol'])]  # Not in supported languages
    )
except FilterError as e:
    # Handle: "Unknown language: 'cobol'"
    # Action: Show supported languages, suggest alternatives
```

## Testing Strategy

### Unit Tests (by component)

1. **MetadataFilter**:
   - Each filter type (LanguageFilter, FilePathFilter, etc.)
   - AND/OR logic correctness
   - Edge cases (empty lists, special characters)
   - 20+ tests

2. **QueryEngine**:
   - search_vector() with mock LanceDB table
   - search_text() with mock embeddings
   - Filter application correctness
   - Pagination edge cases (offset > results, limit=1, etc.)
   - Error cases (missing table, invalid filters)
   - 25+ tests

3. **SearchResult**:
   - Creation and field access
   - String representation
   - 5+ tests

### Integration Tests

1. **Full text search flow**:
   - End-to-end: text → embedding → search → filter → pagination
   - With real embeddings and mock database
   - 10+ tests

2. **Filter combinations**:
   - Multiple filter types together
   - Empty filter results
   - Pagination with filters
   - 8+ tests

### Performance Tests

1. **Latency benchmarks**:
   - Measure end-to-end query time
   - Verify <500ms for 100-1000 result sets
   - 4+ tests

2. **Throughput**:
   - Concurrent queries handling
   - Load testing
   - 3+ tests

### Test Coverage Target

- **Unit tests**: >95% coverage on query module
- **Integration tests**: All major code paths
- **Overall**: >90% code coverage

## Configuration & Tuning

### Search Parameters

```python
# Fetch more results for better filtering
SEARCH_FETCH_MULTIPLIER = 2  # Fetch 2x limit to account for filtered-out results

# Default pagination
DEFAULT_LIMIT = 10
MAX_LIMIT = 100  # Prevent DoS via huge limit requests

# Batch search for multiple queries
BATCH_SEARCH_SIZE = 50
```

### Performance Tuning (Future)

- **Query caching**: Cache popular queries (LRU, 1-hour TTL)
- **Result ranking**: Hybrid scoring (similarity + metadata relevance)
- **Reranking**: Secondary ranking by complexity, popularity
- **Full-text search**: Add BM25 indexing for keyword matching

## Security & Data Privacy

### Input Validation

- Query text: Max 1000 chars, no injection attacks (safe with embeddings)
- Filter values: Validated against allowed values
- Pagination params: Bounds checked (offset ≥ 0, limit ≤ MAX_LIMIT)

### Data Access

- All queries respect existing data access controls (LanceDB permissions)
- No additional authentication required (downstream responsibility)
- Audit logging recommended (queries, filters, results returned)

## Implementation Phases

**Phase 1**: Core query engine + filters (Days 1-2)
- QueryEngine class with search_vector, search_text
- MetadataFilter and concrete implementations
- Exception classes
- 30+ unit tests

**Phase 2**: Integration + performance (Days 3-4)
- Integration tests with real embeddings
- Performance benchmarking
- Error handling validation
- 20+ tests

**Phase 3**: Polish + documentation (Day 5)
- CLI integration
- API documentation
- Usage examples
- Performance report

## Success Criteria

✅ Vector similarity search working with <100ms latency
✅ Metadata filtering (language, file, type, repo) with OR logic
✅ Pagination with offset/limit
✅ Error handling with clear messages
✅ 90%+ test coverage
✅ <500ms P99 latency for typical queries
✅ Handles 1M+ entities efficiently
✅ Clean, documented API

## References

- Issue #10: Data Ingestion Pipeline (bulk insert, deduplication)
- Issue #9: LanceDB Schema Design (code_entities, search_metadata tables)
- CodeEntity, SearchMetadata models
- EmbeddingGenerator (embeddings module)
- LanceDB vector search documentation
