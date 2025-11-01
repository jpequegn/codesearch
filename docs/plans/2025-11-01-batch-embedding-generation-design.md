# Batch Embedding Generation Design

**Date**: 2025-11-01
**Issue**: #7
**Component**: 2.3 - Embedding Pipeline

## Overview

This document specifies the design for batch embedding generation - processing multiple functions and classes efficiently while maintaining a cache to avoid re-computing embeddings. The system generates semantic embeddings for all code entities discovered by the parser, enabling vector-based search capabilities.

## Requirements

- **Primary Goal**: Sequential, clear embedding generation with caching
- **Scale**: Single repository (100-1000 functions)
- **Processing Model**: Sequential (one function at a time)
- **Caching**: In-memory during batch + persistent disk cache
- **Error Handling**: Graceful failure (log and continue)
- **Architecture**: Functional pipeline (orchestrate prepare → check cache → embed → store)

## Architecture

### Core Components

#### BatchEmbeddingGenerator Class

Main orchestrator for processing functions and classes through the embedding pipeline:

```python
class BatchEmbeddingGenerator:
    """Batch process code embeddings with caching."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        text_preparator: TextPreparator,
        cache_dir: str = "~/.codesearch/embeddings"
    ):
        """Initialize with components and cache directory."""
        self.embedding_generator = embedding_generator
        self.text_preparator = text_preparator
        self.cache_dir = cache_dir
        self.cache: Dict[str, EmbeddingCacheEntry] = {}

    def process_functions(
        self, functions: List[Function]
    ) -> Dict[str, Any]:
        """Process list of functions and return embeddings."""

    def process_classes(
        self, classes: List[Class]
    ) -> Dict[str, Any]:
        """Process list of classes and return embeddings."""

    def process_batch(
        self, items: List[Union[Function, Class]]
    ) -> Dict[str, Any]:
        """Process mixed list of functions and classes."""
```

### Processing Pipeline

The batch processing follows this flow:

```
1. Load Cache
   └─> Read cache file from disk
   └─> Populate in-memory cache dict

2. Process Each Item (sequential)
   ├─> Generate cache key: {file_path}:{line_number}
   ├─> Check cache (memory first, then disk)
   ├─> If cache hit: reuse embedding
   └─> If cache miss:
       ├─> Prepare text (via TextPreparator)
       ├─> Generate embedding (via EmbeddingGenerator)
       └─> Store in memory cache

3. Save Cache
   └─> Serialize memory cache to disk JSON

4. Return Results
   └─> Dictionary with summary stats + individual results
```

### Data Flow

```
Parser Output (Functions/Classes)
    ↓
BatchEmbeddingGenerator.process_batch()
    ├─ Load existing cache from disk
    ├─ For each item:
    │  ├─ Check cache for embedding
    │  └─ If missing: TextPreparator → EmbeddingGenerator → cache
    └─ Save cache to disk
    ↓
Embeddings + Metadata Dictionary
    ↓ (next component)
LanceDB Storage
```

## Implementation Details

### Cache Storage Format

Cache stored as JSON file with structure:

```json
{
  "metadata": {
    "model_name": "microsoft/codebert-base",
    "model_version": "1.0",
    "dimensions": 768,
    "created": "2025-11-01T15:00:00Z",
    "updated": "2025-11-01T16:30:00Z"
  },
  "embeddings": {
    "/path/to/file.py:42": {
      "function_name": "calculate_sum",
      "embedding": [0.123, -0.456, ..., 0.789],
      "timestamp": "2025-11-01T15:05:00Z",
      "model_version": "1.0"
    },
    "/path/to/file.py:58": {
      "function_name": "process_data",
      "embedding": [0.789, -0.012, ..., 0.345],
      "timestamp": "2025-11-01T15:10:00Z",
      "model_version": "1.0"
    }
  }
}
```

### Cache Key Strategy

- **Format**: `{file_path}:{line_number}`
- **Example**: `/src/main.py:42` for function at line 42
- **Uniqueness**: Guaranteed unique per function/class (from parser)
- **Deterministic**: Same input always produces same key

### Processing Methods

#### _load_cache()
```python
def _load_cache(self) -> None:
    """Load embeddings from disk cache into memory."""
    try:
        with open(self.cache_path, 'r') as f:
            data = json.load(f)
        self.cache = data.get('embeddings', {})
        self.metadata = data.get('metadata', {})
    except FileNotFoundError:
        self.cache = {}
        self.metadata = self._create_metadata()
    except Exception as e:
        # Log error, continue with empty cache
        logger.warning(f"Failed to load cache: {e}")
        self.cache = {}
```

#### _save_cache()
```python
def _save_cache(self) -> None:
    """Persist in-memory cache to disk."""
    try:
        self.metadata['updated'] = datetime.utcnow().isoformat()
        data = {
            'metadata': self.metadata,
            'embeddings': self.cache
        }
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")
```

#### _embed_and_cache(item)
```python
def _embed_and_cache(
    self, item: Union[Function, Class]
) -> Optional[List[float]]:
    """Prepare, embed, and cache a single item."""
    try:
        # Prepare text
        if isinstance(item, Function):
            text = self.text_preparator.prepare_function(item)
        else:
            text = self.text_preparator.prepare_class(item)

        # Generate embedding
        embedding = self.embedding_generator.embed_code(text)

        # Cache result
        key = self._get_cache_key(item)
        self.cache[key] = {
            'name': item.name,
            'embedding': embedding,
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': self.metadata.get('model_version', '1.0')
        }

        return embedding
    except Exception as e:
        logger.error(f"Failed to embed {item.name}: {e}")
        return None
```

### Result Structure

```python
@dataclass
class EmbeddingBatchResult:
    """Result of batch embedding process."""

    summary: Dict[str, int]  # {total, success, failed, cached, newly_embedded}
    embeddings: Dict[str, Optional[List[float]]]  # {cache_key: embedding}
    errors: Dict[str, str]  # {cache_key: error_message}
    metadata: Dict[str, Any]  # {model_name, model_version, timestamps}
```

Example return value:
```python
{
    "summary": {
        "total": 100,
        "success": 97,
        "failed": 3,
        "cached": 50,
        "newly_embedded": 47
    },
    "embeddings": {
        "/path/file.py:42": [0.123, -0.456, ...],
        "/path/file.py:58": [0.789, -0.012, ...],
        "/path/file.py:100": None  # Failed
    },
    "errors": {
        "/path/file.py:100": "Text preparation failed: syntax error"
    },
    "metadata": {
        "model_name": "microsoft/codebert-base",
        "model_version": "1.0",
        "dimensions": 768,
        "processed_at": "2025-11-01T16:30:00Z"
    }
}
```

## Error Handling & Recovery

### Graceful Failure Pattern

For each function/class, if any step fails:
1. **Log** the error with context (name, file, line number)
2. **Track** the failure (increment failed counter)
3. **Mark** result as failed: `embeddings[key] = None`
4. **Continue** processing remaining items

**Never crash the batch due to single item failure.**

### Error Categories

- **Text Preparation Failure**: Docstring extraction issues, malformed code
  - Action: Log warning, skip item
- **Embedding Generation Failure**: Model errors, OOM
  - Action: Log error, skip item
- **Cache I/O Failure**: Permission denied, disk full
  - Action: Log warning, continue with in-memory only
- **Invalid Input**: None function, missing attributes
  - Action: Log error, skip item

## Integration Points

### With TextPreparator (Issue #6)
- Calls `text_preparator.prepare_function()` and `prepare_class()`
- Uses prepared text for embedding generation
- Handles TextPreparator exceptions gracefully

### With EmbeddingGenerator (Issue #5)
- Calls `embedding_generator.embed_code(text)`
- Expects 768-dimensional float list
- Handles embedding exceptions gracefully

### With Parser (Issues #1-4)
- Receives `Function` and `Class` objects
- Uses `file_path`, `line_number`, `name`, `docstring`, `source_code`
- Generates unique cache keys from these fields

### With LanceDB (Issue #9+)
- Outputs embeddings dictionary ready for vector storage
- Metadata includes model version for schema compatibility
- Cache key format compatible with function/class identification

## Testing Strategy

### Unit Tests

**Basic Functionality:**
- `test_process_single_function` - Embed one function
- `test_process_multiple_functions` - Batch of functions
- `test_process_classes` - Batch of classes
- `test_process_mixed_batch` - Functions and classes together

**Caching:**
- `test_cache_hit` - Reuses cached embedding
- `test_cache_miss` - Computes and stores new embedding
- `test_cache_persistence` - Loads/saves to disk correctly
- `test_cache_hit_tracking` - Summary counts cache hits/misses

**Error Handling:**
- `test_error_text_preparation` - Continues on text prep failure
- `test_error_embedding_generation` - Continues on embedding failure
- `test_error_invalid_input` - Handles None input gracefully
- `test_batch_partial_failure` - Failed items don't stop batch

**Metadata:**
- `test_metadata_tracking` - Timestamps and versions recorded
- `test_result_summary_accuracy` - Summary counts match actual results
- `test_model_version_detection` - Detects model version mismatch

### Integration Tests

- `test_end_to_end_parser_to_embeddings` - Parser output → batch embeddings
- `test_real_codebase_p3` - Process 100+ functions from P3 codebase
- `test_embedding_consistency` - Same input always produces same embedding

### Edge Cases

- Empty input list
- Single function batch
- Functions with no docstrings
- Very long function bodies (truncation tested)
- Cache corruption (graceful fallback)
- Missing cache file
- Unicode in function names

## Success Criteria

✓ BatchEmbeddingGenerator class created with all required methods
✓ Caching system working (in-memory + disk persistence)
✓ Sequential batch processing completing without crashes
✓ Failed items logged but don't stop batch
✓ Summary statistics accurate (total, success, failed, cached, newly_embedded)
✓ All tests passing (20+ unit + integration tests)
✓ >90% code coverage on batch_embedding_generator.py
✓ Real-world validation on P3 codebase (100+ functions)
✓ Performance improvement from caching verified
✓ Ready for LanceDB integration (Issue #9)

## Files to Create

- `codesearch/embeddings/batch_generator.py` - BatchEmbeddingGenerator implementation
- `tests/test_batch_embedding_generator.py` - Comprehensive test suite
- Cache directory: `~/.codesearch/embeddings/cache.json` - Persistent storage

## Next Steps

1. Implement BatchEmbeddingGenerator class with all methods
2. Write comprehensive test suite (20+ tests)
3. Validate on real P3 codebase
4. Commit PR for review
5. Proceed to Issue #8: Embedding Quality Assurance (metrics, validation)
6. Then Issue #9: LanceDB Schema Design
