# Python API Documentation - Codesearch

For programmatic access to Codesearch functionality, use the Python API. All modules are importable and designed for integration into Python applications.

## Installation

```bash
pip install -e .
```

## Core Modules

### Query Module (`codesearch.query`)

Search and retrieve code entities from the database.

#### QueryEngine

Main interface for querying code entities.

```python
from codesearch.query.engine import QueryEngine

# Initialize the query engine
engine = QueryEngine(db_path="~/.codesearch")

# Search by pattern (semantic search)
results = engine.search_pattern(
    query="function that validates email",
    language="python",
    limit=10,
    offset=0,
    threshold=0.0
)

# Find similar functions
similar = engine.find_similar(
    entity_name="my_function",
    language="python",
    limit=10,
    include_self=False
)

# Get entity dependencies
deps = engine.get_dependencies(
    entity_id="function_id",
    direction="both"  # "callers", "callees", or "both"
)
```

#### SearchResult

Represents a single search result.

```python
from codesearch.query.models import SearchResult

result = SearchResult(
    id="entity_abc123",
    name="validate_email",
    type="function",
    language="python",
    file="utils/validators.py",
    line_start=42,
    line_end=55,
    signature="def validate_email(email: str) -> bool",
    docstring="Validate email format",
    score=0.92
)

# Access properties
print(result.name)          # "validate_email"
print(result.signature)     # "def validate_email(email: str) -> bool"
print(result.score)         # 0.92
```

#### MetadataFilter

Filter search results by metadata.

```python
from codesearch.query.models import MetadataFilter, LanguageFilter, FileFilter

# Single language filter
lang_filter = LanguageFilter(languages=["python"])

# Multiple languages (OR logic within type)
lang_filter = LanguageFilter(languages=["python", "typescript"])

# File path filter
file_filter = FileFilter(patterns=["src/**/*.py"])

# Repository filter
repo_filter = RepositoryFilter(repo_ids=["repo-1", "repo-2"])

# Combine filters (AND logic between types)
results = engine.search_pattern(
    query="validation",
    filters=[lang_filter, file_filter]
)
```

---

### Indexing Module (`codesearch.indexing`)

Index code repositories for semantic search.

#### DataIngestionPipeline

Orchestrates repository indexing (currently a stub, see Issue #10).

```python
from codesearch.indexing.pipeline import DataIngestionPipeline
from codesearch.indexing.validators import IngestionValidator

# Initialize pipeline
pipeline = DataIngestionPipeline(
    db_path="~/.codesearch",
    batch_size=32,
    workers=4
)

# Index a repository
result = pipeline.index_repository(
    repo_path="/path/to/repo",
    repo_id="my-repo",
    languages=["python"],
    incremental=True
)

# Access results
print(result.entities_indexed)  # Number of entities indexed
print(result.failures)          # List of failed entities
print(result.skipped)           # List of skipped (unchanged) entities
```

#### IngestionValidator

Validates code entities before insertion (planned for Issue #10).

```python
from codesearch.indexing.validators import IngestionValidator
from codesearch.models import CodeEntity

validator = IngestionValidator()

entity = CodeEntity(
    name="my_function",
    type="function",
    language="python",
    file="utils.py",
    signature="def my_function(x: int) -> int"
)

# Validate entity
is_valid = validator.validate(entity)

# Get validation errors
errors = validator.get_errors(entity)
for error in errors:
    print(f"Validation error: {error}")
```

---

### Models Module (`codesearch.models`)

Data models for code entities and relationships.

#### CodeEntity

Represents a code entity (function, class, method).

```python
from codesearch.models import CodeEntity

entity = CodeEntity(
    entity_id="abc123",
    name="process_request",
    type="function",
    language="python",
    signature="async def process_request(request: Request) -> Response",
    docstring="Process incoming HTTP request",
    source_file="handlers/request.py",
    start_line=42,
    end_line=65,
    embedding=None,  # Set by embedding pipeline
    repository_id="repo-1",
    created_at=datetime.now(),
    updated_at=datetime.now()
)

# Access properties
print(entity.name)              # "process_request"
print(entity.type)              # "function"
print(entity.source_file)       # "handlers/request.py"
```

#### CodeRelationship

Represents a call relationship between entities.

```python
from codesearch.models import CodeRelationship

relationship = CodeRelationship(
    relationship_id="rel_123",
    caller_id="entity_abc123",
    callee_id="entity_def456",
    call_count=5,
    call_locations=[100, 102, 105, 110, 115],
    relationship_type="calls",
    repository_id="repo-1",
    created_at=datetime.now()
)

# Access properties
print(relationship.caller_id)       # "entity_abc123"
print(relationship.callee_id)       # "entity_def456"
print(relationship.call_count)      # 5
```

---

### Embeddings Module (`codesearch.embeddings`)

Generate semantic embeddings for code.

#### EmbeddingGenerator

Generate embeddings for code entities.

```python
from codesearch.embeddings.generator import EmbeddingGenerator

# Initialize with default model
generator = EmbeddingGenerator()

# Or specify custom model
generator = EmbeddingGenerator(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Generate embedding for code snippet
code_snippet = "def validate_email(email: str) -> bool: ..."
embedding = generator.embed_code(code_snippet)
print(embedding.shape)  # (384,) or (768,) depending on model

# Generate embeddings in batch
snippets = ["def func1(): ...", "def func2(): ...", "def func3(): ..."]
embeddings = generator.embed_batch(snippets)
print(embeddings.shape)  # (3, 384)

# Generate embedding for query
query = "function that validates email"
query_embedding = generator.embed_query(query)
```

#### Available Models

```python
# Compact model (recommended for most use cases)
generator = EmbeddingGenerator(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# Output: 384-dimensional vectors
# Size: ~22 MB

# Large model (more accurate, slower)
generator = EmbeddingGenerator(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
# Output: 768-dimensional vectors
# Size: ~438 MB

# Custom model
generator = EmbeddingGenerator(
    model_name="path/to/custom/model"
)
```

---

### Caching Module (`codesearch.caching`)

Optimize performance with caching (Issue #5 - Complete).

#### CacheManager

Manage cached embeddings and query results.

```python
from codesearch.caching.manager import CacheManager

# Initialize cache
cache = CacheManager(cache_dir="~/.codesearch/cache")

# Cache an embedding
embedding_key = "snippet_abc123"
embedding_vector = [0.1, 0.2, 0.3, ...]  # 384-dim vector
cache.set_embedding(embedding_key, embedding_vector)

# Retrieve cached embedding
cached = cache.get_embedding(embedding_key)

# Cache search results
query_key = "pattern:email_validation"
results = [...]  # List of SearchResult objects
cache.set_query_result(query_key, results)

# Clear cache
cache.clear()
cache.clear_embeddings()
cache.clear_queries()

# Get cache statistics
stats = cache.get_stats()
print(f"Cache size: {stats['size_mb']} MB")
print(f"Cached embeddings: {stats['embedding_count']}")
print(f"Cached queries: {stats['query_count']}")
```

---

## Common Workflows

### Workflow 1: Search for Validation Functions

```python
from codesearch.query.engine import QueryEngine
from codesearch.query.models import LanguageFilter

# Initialize engine
engine = QueryEngine(db_path="~/.codesearch")

# Search for validation patterns
results = engine.search_pattern(
    query="function that validates input",
    filters=[LanguageFilter(languages=["python"])],
    limit=20
)

# Process results
for result in results:
    print(f"{result.name} ({result.file}:{result.line_start})")
    if result.docstring:
        print(f"  Doc: {result.docstring}")
    print(f"  Score: {result.score:.2f}\n")
```

### Workflow 2: Find Similar Implementation

```python
from codesearch.query.engine import QueryEngine

engine = QueryEngine(db_path="~/.codesearch")

# Find similar to a known function
similar = engine.find_similar(
    entity_name="database_query",
    limit=10,
    threshold=0.85
)

# Group by similarity
for result in similar:
    print(f"{result.name} - {result.score:.2f} ({result.file})")
```

### Workflow 3: Analyze Code Dependencies

```python
from codesearch.query.engine import QueryEngine

engine = QueryEngine(db_path="~/.codesearch")

# Get dependencies for a function
deps = engine.get_dependencies(
    entity_id="process_request_id",
    direction="both"
)

print("Functions that call this:")
for caller in deps.callers:
    print(f"  - {caller.name} ({caller.file}:{caller.line_start})")

print("\nFunctions this calls:")
for callee in deps.callees:
    print(f"  - {callee.name} ({callee.file}:{callee.line_start})")
```

### Workflow 4: Index a New Repository

```python
from codesearch.indexing.pipeline import DataIngestionPipeline

# Initialize pipeline
pipeline = DataIngestionPipeline(
    db_path="~/.codesearch",
    workers=4
)

# Index repository
result = pipeline.index_repository(
    repo_path="/path/to/new/repo",
    repo_id="new-repo",
    languages=["python", "typescript"],
    incremental=True
)

# Check results
print(f"Indexed: {result.entities_indexed}")
print(f"Failed: {len(result.failures)}")
print(f"Skipped: {len(result.skipped)}")

# Handle failures
for failure in result.failures:
    print(f"Failed to index {failure.file}: {failure.error}")
```

### Workflow 5: Cache Embeddings for Performance

```python
from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.caching.manager import CacheManager

# Setup
generator = EmbeddingGenerator()
cache = CacheManager()

# Check cache first
code_snippet = "def my_function(): ..."
cache_key = hash(code_snippet)

embedding = cache.get_embedding(cache_key)
if embedding is None:
    # Not in cache, generate
    embedding = generator.embed_code(code_snippet)
    # Store in cache
    cache.set_embedding(cache_key, embedding)

# Use embedding
print(f"Embedding shape: {embedding.shape}")
```

---

## Error Handling

All API methods raise specific exceptions for error handling:

```python
from codesearch.query.engine import QueryEngine
from codesearch.exceptions import (
    DatabaseError,
    QueryError,
    ConfigurationError
)

engine = QueryEngine(db_path="~/.codesearch")

try:
    results = engine.search_pattern(
        query="validation",
        limit=10
    )
except ConfigurationError as e:
    print(f"Configuration problem: {e}")
except DatabaseError as e:
    print(f"Database error: {e}")
except QueryError as e:
    print(f"Query error: {e}")
```

### Exception Types

| Exception | Cause | Handling |
|-----------|-------|----------|
| `ConfigurationError` | Invalid configuration or arguments | Check config and arguments |
| `DatabaseError` | Database connection or schema error | Verify database path and permissions |
| `QueryError` | Invalid query or entity not found | Check query syntax or entity name |
| `EmbeddingError` | Embedding generation failed | Verify model availability |
| `ValidationError` | Data validation failed | Check entity data |

---

## Type Hints

All APIs use type hints for IDE support:

```python
from typing import List, Optional
from codesearch.query.models import SearchResult

def process_results(results: List[SearchResult]) -> None:
    """Process search results."""
    for result in results:
        name: str = result.name
        score: float = result.score
        print(f"{name}: {score:.2f}")
```

---

## Configuration

### Via Environment Variables

```bash
export CODESEARCH_DB_PATH=~/.codesearch
export CODESEARCH_LANGUAGE=python
export CODESEARCH_OUTPUT_FORMAT=json
```

### Via Python Code

```python
import os

os.environ["CODESEARCH_DB_PATH"] = "/custom/path/.codesearch"
os.environ["CODESEARCH_LANGUAGE"] = "typescript"

# Now initialize API with configured values
from codesearch.query.engine import QueryEngine
engine = QueryEngine()  # Uses environment settings
```

---

## Performance Tips

### 1. Batch Operations

```python
# Batch embedding generation (faster than individual)
snippets = ["code1", "code2", "code3", ...]
embeddings = generator.embed_batch(snippets)
```

### 2. Use Caching

```python
# Cache frequently searched queries
cache.set_query_result("pattern:email_validation", results)
cached_results = cache.get_query_result("pattern:email_validation")
```

### 3. Pagination for Large Results

```python
# Instead of fetching all results at once
offset = 0
limit = 50

while True:
    results = engine.search_pattern(
        query="validation",
        offset=offset,
        limit=limit
    )

    if not results:
        break

    # Process batch
    process_batch(results)

    offset += limit
```

### 4. Language Filtering

```python
# Filter early to reduce search space
results = engine.search_pattern(
    query="validation",
    filters=[LanguageFilter(languages=["python"])],
    limit=10
)
```

---

## See Also

- [Architecture](ARCHITECTURE.md) - System design
- [CLI Reference](CLI.md) - Command-line interface
- [Contributing](../CONTRIBUTING.md) - Development guide
- [Troubleshooting](TROUBLESHOOTING.md) - Debugging help
