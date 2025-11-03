# Issue #7: Batch Embedding Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task with code review between tasks.

**Goal:** Implement BatchEmbeddingGenerator to process multiple code entities through the embedding pipeline with efficient caching and graceful error handling.

**Architecture:** Functional pipeline orchestrator that loads cache, processes items sequentially (prepare → check cache → embed → store), saves cache, and returns detailed results with summary statistics.

**Tech Stack:** Python 3.9+, Pydantic for result types, JSON for persistent caching, pytest for testing.

---

## Task 1: Create BatchEmbeddingGenerator Basic Structure

**Files:**
- Create: `codesearch/embeddings/batch_generator.py`
- Modify: `codesearch/embeddings/__init__.py`
- Create: `tests/test_batch_embedding_generator.py`

**Step 1: Write the failing test**

```python
# tests/test_batch_embedding_generator.py
"""Tests for batch embedding generator."""

import pytest
from codesearch.models import Function, Class
from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.embeddings.text_preparator import TextPreparator
from codesearch.embeddings.batch_generator import BatchEmbeddingGenerator


def test_batch_generator_initialization():
    """Test creating a BatchEmbeddingGenerator."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    assert batch_gen is not None
    assert batch_gen.embedding_generator is not None
    assert batch_gen.text_preparator is not None
    assert isinstance(batch_gen.cache, dict)


def test_batch_generator_has_required_methods():
    """Test that BatchEmbeddingGenerator has all required methods."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    # Verify all methods exist
    assert hasattr(batch_gen, 'process_functions')
    assert hasattr(batch_gen, 'process_classes')
    assert hasattr(batch_gen, 'process_batch')
    assert hasattr(batch_gen, '_load_cache')
    assert hasattr(batch_gen, '_save_cache')
    assert hasattr(batch_gen, '_get_cache_key')
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/julienpequegnot/Code/codesearch/.worktrees/issue-7-batch-embedding
python3 -m pytest tests/test_batch_embedding_generator.py -v
```

Expected: FAIL - "ModuleNotFoundError: No module named 'codesearch.embeddings.batch_generator'"

**Step 3: Write minimal implementation**

```python
# codesearch/embeddings/batch_generator.py
"""Batch embedding generation with caching."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from codesearch.models import Function, Class
from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.embeddings.text_preparator import TextPreparator


class BatchEmbeddingGenerator:
    """Generate embeddings for multiple functions/classes with caching."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        text_preparator: TextPreparator,
        cache_dir: str = "~/.codesearch/embeddings"
    ):
        """Initialize batch generator with components and cache directory."""
        self.embedding_generator = embedding_generator
        self.text_preparator = text_preparator
        self.cache_dir = os.path.expanduser(cache_dir)
        self.cache: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.cache_path = os.path.join(self.cache_dir, "cache.json")

    def process_functions(
        self, functions: List[Function]
    ) -> Dict[str, Any]:
        """Process list of functions and return embeddings."""
        return self.process_batch(functions)

    def process_classes(
        self, classes: List[Class]
    ) -> Dict[str, Any]:
        """Process list of classes and return embeddings."""
        return self.process_batch(classes)

    def process_batch(
        self, items: List[Union[Function, Class]]
    ) -> Dict[str, Any]:
        """Process mixed list of functions and classes."""
        # Placeholder - will implement in next tasks
        return {
            "summary": {"total": 0, "success": 0, "failed": 0, "cached": 0, "newly_embedded": 0},
            "embeddings": {},
            "errors": {},
            "metadata": {}
        }

    def _load_cache(self) -> None:
        """Load embeddings from disk cache into memory."""
        pass

    def _save_cache(self) -> None:
        """Persist in-memory cache to disk."""
        pass

    def _get_cache_key(self, item: Union[Function, Class]) -> str:
        """Generate unique cache key for function/class."""
        return f"{item.file_path}:{item.line_number}"
```

Update `__init__.py`:

```python
# codesearch/embeddings/__init__.py
"""Embedding generation module."""

from codesearch.embeddings.batch_generator import BatchEmbeddingGenerator
from codesearch.embeddings.text_preparator import TextPreparator
from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.models import EmbeddingModel

__all__ = ["BatchEmbeddingGenerator", "TextPreparator", "EmbeddingGenerator", "EmbeddingModel"]
```

**Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py::test_batch_generator_initialization -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_batch_generator_has_required_methods -v
```

Expected: PASS (2/2 tests)

**Step 5: Commit**

```bash
git add codesearch/embeddings/batch_generator.py codesearch/embeddings/__init__.py tests/test_batch_embedding_generator.py
git commit -m "feat: Create BatchEmbeddingGenerator basic structure (Task 1)"
```

---

## Task 2: Implement Cache Loading and Persistence

**Files:**
- Modify: `codesearch/embeddings/batch_generator.py`
- Modify: `tests/test_batch_embedding_generator.py`

**Step 1: Write the failing tests**

Add to `tests/test_batch_embedding_generator.py`:

```python
def test_load_cache_creates_metadata():
    """Test that _load_cache initializes metadata."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    batch_gen._load_cache()

    # Should have metadata even if cache file doesn't exist
    assert batch_gen.metadata is not None
    assert isinstance(batch_gen.metadata, dict)


def test_save_cache_creates_file():
    """Test that _save_cache persists cache to disk."""
    import tempfile
    import json

    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    with tempfile.TemporaryDirectory() as tmpdir:
        batch_gen = BatchEmbeddingGenerator(
            generator, preparator,
            cache_dir=tmpdir
        )

        # Add something to cache
        batch_gen.cache["/test.py:1"] = {
            "name": "test_func",
            "embedding": [0.1, 0.2, 0.3],
            "timestamp": "2025-11-01T00:00:00Z",
            "model_version": "1.0"
        }
        batch_gen.metadata = {
            "model_name": "codebert-base",
            "model_version": "1.0"
        }

        batch_gen._save_cache()

        # Verify file exists
        assert os.path.exists(batch_gen.cache_path)

        # Verify file contains expected data
        with open(batch_gen.cache_path, 'r') as f:
            data = json.load(f)

        assert "embeddings" in data
        assert "/test.py:1" in data["embeddings"]


def test_load_existing_cache():
    """Test that _load_cache reads existing cache file."""
    import tempfile
    import json

    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create cache file
        cache_path = os.path.join(tmpdir, "cache.json")
        cache_data = {
            "metadata": {
                "model_name": "codebert-base",
                "model_version": "1.0"
            },
            "embeddings": {
                "/existing.py:10": {
                    "name": "existing_func",
                    "embedding": [0.5, 0.6, 0.7],
                    "timestamp": "2025-11-01T00:00:00Z",
                    "model_version": "1.0"
                }
            }
        }

        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

        batch_gen = BatchEmbeddingGenerator(
            generator, preparator,
            cache_dir=tmpdir
        )

        batch_gen._load_cache()

        # Verify cache loaded
        assert "/existing.py:10" in batch_gen.cache
        assert batch_gen.cache["/existing.py:10"]["name"] == "existing_func"
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py::test_load_cache_creates_metadata -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_save_cache_creates_file -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_load_existing_cache -v
```

Expected: FAIL (3 failing tests)

**Step 3: Implement cache loading and saving**

Update `codesearch/embeddings/batch_generator.py`:

```python
def _load_cache(self) -> None:
    """Load embeddings from disk cache into memory."""
    try:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
            self.cache = data.get('embeddings', {})
            self.metadata = data.get('metadata', {})
        else:
            self.cache = {}
            self.metadata = self._create_metadata()
    except Exception as e:
        # Log error, continue with empty cache
        print(f"Warning: Failed to load cache: {e}")
        self.cache = {}
        self.metadata = self._create_metadata()

def _save_cache(self) -> None:
    """Persist in-memory cache to disk."""
    try:
        self.metadata['updated'] = datetime.utcnow().isoformat() + 'Z'
        data = {
            'metadata': self.metadata,
            'embeddings': self.cache
        }
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error: Failed to save cache: {e}")

def _create_metadata(self) -> Dict[str, Any]:
    """Create initial metadata."""
    return {
        "model_name": self.embedding_generator.model_config.name,
        "model_version": "1.0",
        "dimensions": self.embedding_generator.model_config.dimensions,
        "created": datetime.utcnow().isoformat() + 'Z',
        "updated": datetime.utcnow().isoformat() + 'Z'
    }
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py::test_load_cache_creates_metadata -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_save_cache_creates_file -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_load_existing_cache -v
```

Expected: PASS (3/3 tests)

**Step 5: Commit**

```bash
git add codesearch/embeddings/batch_generator.py tests/test_batch_embedding_generator.py
git commit -m "feat: Implement cache loading and persistence (Task 2)"
```

---

## Task 3: Implement Core Batch Processing Logic

**Files:**
- Modify: `codesearch/embeddings/batch_generator.py`
- Modify: `tests/test_batch_embedding_generator.py`

**Step 1: Write the failing tests**

Add to `tests/test_batch_embedding_generator.py`:

```python
def test_process_single_function():
    """Test processing a single function."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    func = Function(
        name="add",
        file_path="/test.py",
        language="python",
        source_code="def add(a, b):\n    return a + b",
        docstring="Add two numbers.",
        line_number=1,
    )

    result = batch_gen.process_functions([func])

    # Should return dict with summary and embeddings
    assert "summary" in result
    assert "embeddings" in result
    assert result["summary"]["total"] == 1
    assert result["summary"]["success"] == 1
    assert "/test.py:1" in result["embeddings"]


def test_process_multiple_functions():
    """Test processing multiple functions."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    functions = [
        Function(
            name="add", file_path="/test.py", language="python",
            source_code="def add(a, b):\n    return a + b",
            docstring="Add numbers.", line_number=1,
        ),
        Function(
            name="sub", file_path="/test.py", language="python",
            source_code="def sub(a, b):\n    return a - b",
            docstring="Subtract numbers.", line_number=5,
        ),
    ]

    result = batch_gen.process_functions(functions)

    assert result["summary"]["total"] == 2
    assert result["summary"]["success"] == 2
    assert len(result["embeddings"]) == 2


def test_process_classes():
    """Test processing classes."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    cls = Class(
        name="Calculator",
        file_path="/test.py",
        language="python",
        source_code="class Calculator:\n    pass",
        docstring="A calculator class.",
        line_number=10,
    )

    result = batch_gen.process_classes([cls])

    assert result["summary"]["total"] == 1
    assert result["summary"]["success"] == 1
    assert "/test.py:10" in result["embeddings"]
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py::test_process_single_function -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_process_multiple_functions -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_process_classes -v
```

Expected: FAIL (tests fail because process_batch returns placeholder)

**Step 3: Implement batch processing logic**

Update `codesearch/embeddings/batch_generator.py`:

```python
def process_batch(
    self, items: List[Union[Function, Class]]
) -> Dict[str, Any]:
    """Process mixed list of functions and classes."""
    # Load existing cache
    self._load_cache()

    summary = {
        "total": len(items),
        "success": 0,
        "failed": 0,
        "cached": 0,
        "newly_embedded": 0
    }

    embeddings: Dict[str, Optional[List[float]]] = {}
    errors: Dict[str, str] = {}

    for item in items:
        key = self._get_cache_key(item)

        # Check if in cache
        if key in self.cache:
            embeddings[key] = self.cache[key].get("embedding")
            summary["cached"] += 1
            summary["success"] += 1
        else:
            # Embed new item
            embedding = self._embed_and_cache(item)
            embeddings[key] = embedding

            if embedding is not None:
                summary["newly_embedded"] += 1
                summary["success"] += 1
            else:
                summary["failed"] += 1
                errors[key] = f"Failed to embed {item.name}"

    # Save cache
    self._save_cache()

    return {
        "summary": summary,
        "embeddings": embeddings,
        "errors": errors,
        "metadata": self.metadata
    }

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
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'model_version': self.metadata.get('model_version', '1.0')
        }

        return embedding
    except Exception as e:
        print(f"Error embedding {item.name}: {e}")
        return None
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py::test_process_single_function -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_process_multiple_functions -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_process_classes -v
```

Expected: PASS (3/3 tests)

**Step 5: Commit**

```bash
git add codesearch/embeddings/batch_generator.py tests/test_batch_embedding_generator.py
git commit -m "feat: Implement core batch processing logic (Task 3)"
```

---

## Task 4: Implement Caching and Hit Tracking

**Files:**
- Modify: `codesearch/embeddings/batch_generator.py`
- Modify: `tests/test_batch_embedding_generator.py`

**Step 1: Write the failing tests**

Add to `tests/test_batch_embedding_generator.py`:

```python
def test_cache_hit():
    """Test that cached embeddings are reused."""
    import tempfile

    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    with tempfile.TemporaryDirectory() as tmpdir:
        batch_gen = BatchEmbeddingGenerator(generator, preparator, cache_dir=tmpdir)

        func = Function(
            name="test", file_path="/test.py", language="python",
            source_code="def test(): pass",
            docstring="Test function.", line_number=1,
        )

        # First run - should compute
        result1 = batch_gen.process_functions([func])
        assert result1["summary"]["newly_embedded"] == 1
        assert result1["summary"]["cached"] == 0

        # Second run - should use cache
        batch_gen2 = BatchEmbeddingGenerator(generator, preparator, cache_dir=tmpdir)
        result2 = batch_gen2.process_functions([func])
        assert result2["summary"]["cached"] == 1
        assert result2["summary"]["newly_embedded"] == 0

        # Embeddings should match
        assert result1["embeddings"]["/test.py:1"] == result2["embeddings"]["/test.py:1"]


def test_mixed_cache_hit_miss():
    """Test batch with both cached and new embeddings."""
    import tempfile

    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    with tempfile.TemporaryDirectory() as tmpdir:
        batch_gen = BatchEmbeddingGenerator(generator, preparator, cache_dir=tmpdir)

        func1 = Function(
            name="cached", file_path="/test.py", language="python",
            source_code="def cached(): pass",
            docstring="Cached.", line_number=1,
        )

        # First run
        batch_gen.process_functions([func1])

        # Second run with new function
        func2 = Function(
            name="new", file_path="/test.py", language="python",
            source_code="def new(): pass",
            docstring="New.", line_number=10,
        )

        batch_gen2 = BatchEmbeddingGenerator(generator, preparator, cache_dir=tmpdir)
        result = batch_gen2.process_functions([func1, func2])

        assert result["summary"]["total"] == 2
        assert result["summary"]["cached"] == 1
        assert result["summary"]["newly_embedded"] == 1
        assert result["summary"]["success"] == 2
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py::test_cache_hit -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_mixed_cache_hit_miss -v
```

Expected: FAIL (cache not persisting correctly)

**Step 3: Verify and fix cache implementation**

The cache implementation from Task 2 should handle this. Run the tests again to confirm the code works.

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py::test_cache_hit -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_mixed_cache_hit_miss -v
```

Expected: PASS (2/2 tests)

**Step 5: Commit**

```bash
git add tests/test_batch_embedding_generator.py
git commit -m "feat: Add caching and cache hit tracking tests (Task 4)"
```

---

## Task 5: Implement Error Handling and Graceful Failures

**Files:**
- Modify: `codesearch/embeddings/batch_generator.py`
- Modify: `tests/test_batch_embedding_generator.py`

**Step 1: Write the failing tests**

Add to `tests/test_batch_embedding_generator.py`:

```python
def test_batch_continues_on_error():
    """Test that batch processing continues when one item fails."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    # Create function with invalid docstring that would cause issues
    functions = [
        Function(
            name="good", file_path="/test.py", language="python",
            source_code="def good(): return 1",
            docstring="Good function.", line_number=1,
        ),
        # This should fail or be handled gracefully
        Function(
            name="empty", file_path="/test.py", language="python",
            source_code="",  # Empty code
            docstring=None, line_number=5,
        ),
        Function(
            name="also_good", file_path="/test.py", language="python",
            source_code="def also_good(): return 2",
            docstring="Also good.", line_number=10,
        ),
    ]

    result = batch_gen.process_functions(functions)

    # Should have processed 3, succeeded on at least 2
    assert result["summary"]["total"] == 3
    assert result["summary"]["success"] >= 2
    # Batch should not crash
    assert "summary" in result


def test_empty_batch():
    """Test processing empty batch."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    result = batch_gen.process_functions([])

    assert result["summary"]["total"] == 0
    assert result["summary"]["success"] == 0
    assert len(result["embeddings"]) == 0
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py::test_batch_continues_on_error -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_empty_batch -v
```

Expected: Tests should pass (error handling already in _embed_and_cache)

**Step 3: Enhance error handling if needed**

Update `_embed_and_cache` if necessary to handle edge cases:

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

        # Handle empty text
        if not text or not text.strip():
            print(f"Warning: Empty text for {item.name}")
            return None

        # Generate embedding
        embedding = self.embedding_generator.embed_code(text)

        # Cache result
        key = self._get_cache_key(item)
        self.cache[key] = {
            'name': item.name,
            'embedding': embedding,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'model_version': self.metadata.get('model_version', '1.0')
        }

        return embedding
    except Exception as e:
        print(f"Error embedding {item.name}: {e}")
        return None
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py::test_batch_continues_on_error -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_empty_batch -v
```

Expected: PASS (2/2 tests)

**Step 5: Commit**

```bash
git add codesearch/embeddings/batch_generator.py tests/test_batch_embedding_generator.py
git commit -m "feat: Implement error handling and graceful failures (Task 5)"
```

---

## Task 6: Integration Testing with EmbeddingGenerator and TextPreparator

**Files:**
- Modify: `tests/test_batch_embedding_generator.py`

**Step 1: Write the failing tests**

Add to `tests/test_batch_embedding_generator.py`:

```python
def test_integration_with_full_pipeline():
    """Test complete pipeline: Parser → TextPreparator → BatchEmbeddingGenerator."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    functions = [
        Function(
            name="calculate",
            file_path="/math.py",
            language="python",
            source_code="def calculate(a, b):\n    # TODO: add more operations\n    return a + b",
            docstring="Calculate sum of two numbers.",
            line_number=1,
        ),
        Function(
            name="process",
            file_path="/math.py",
            language="python",
            source_code="def process(data):\n    # important comment\n    return sum(data)",
            docstring="Process data list.",
            line_number=10,
        ),
    ]

    result = batch_gen.process_functions(functions)

    # Verify full pipeline worked
    assert result["summary"]["success"] == 2
    assert len(result["embeddings"]) == 2

    # Verify embeddings are valid
    for key, embedding in result["embeddings"].items():
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 768


def test_integration_end_to_end_with_real_models():
    """Test end-to-end with real EmbeddingGenerator and TextPreparator."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    # Create realistic functions
    functions = [
        Function(
            name="validate_input",
            file_path="/validators.py",
            language="python",
            source_code="""def validate_input(data):
    # TODO: add type checking
    if not data:
        raise ValueError("Empty data")
    return True""",
            docstring="Validate input data before processing.",
            line_number=1,
        ),
        Function(
            name="transform_data",
            file_path="/validators.py",
            language="python",
            source_code="""def transform_data(items):
    \"\"\"Transform items into output format.\"\"\"
    return [item.upper() for item in items]""",
            docstring="Transform items to uppercase.",
            line_number=10,
        ),
    ]

    result = batch_gen.process_functions(functions)

    # All should succeed
    assert result["summary"]["total"] == 2
    assert result["summary"]["success"] == 2
    assert result["summary"]["failed"] == 0

    # Verify embeddings are distinct
    emb1 = result["embeddings"]["/validators.py:1"]
    emb2 = result["embeddings"]["/validators.py:10"]
    assert emb1 != emb2
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py::test_integration_with_full_pipeline -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_integration_end_to_end_with_real_models -v
```

Expected: Tests may pass if implementation is correct

**Step 3: Fix any issues**

If tests fail, verify EmbeddingGenerator and TextPreparator are working correctly by checking their output.

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py::test_integration_with_full_pipeline -v
python3 -m pytest tests/test_batch_embedding_generator.py::test_integration_end_to_end_with_real_models -v
```

Expected: PASS (2/2 tests)

**Step 5: Commit**

```bash
git add tests/test_batch_embedding_generator.py
git commit -m "feat: Add integration testing with EmbeddingGenerator and TextPreparator (Task 6)"
```

---

## Task 7: Real Codebase Validation and PR Creation

**Files:**
- Modify: `tests/test_batch_embedding_generator.py`

**Step 1: Write real codebase validation test**

Add to `tests/test_batch_embedding_generator.py`:

```python
def test_real_codebase_p3_validation():
    """Validate batch embedding on real P3 codebase."""
    import pytest
    from pathlib import Path
    from codesearch.parser import PythonParser

    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)
    parser = PythonParser()

    # Test on P3 codebase
    p3_path = Path("/Users/julienpequegnot/Code/parakeet-podcast-processor/src")

    if not p3_path.exists():
        pytest.skip("P3 codebase not available for testing")

    python_files = list(p3_path.rglob("*.py"))
    assert len(python_files) > 0, "No Python files found in P3"

    total_functions = 0
    total_embedded = 0

    for py_file in python_files[:10]:  # Test first 10 files
        try:
            functions, classes = parser.parse(str(py_file))
            total_functions += len(functions)

            result = batch_gen.process_functions(functions)
            total_embedded += result["summary"]["success"]
        except Exception as e:
            print(f"Warning: Failed to process {py_file}: {e}")

    # Verify reasonable success
    assert total_functions > 0
    assert total_embedded > 0
    assert total_embedded >= total_functions * 0.8  # At least 80% success

    print(f"\nP3 Validation: {total_functions} functions, {total_embedded} embedded")
```

**Step 2: Run full test suite**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py -v --tb=short
```

Expected: All tests passing (20+ tests total)

**Step 3: Run full project test suite**

```bash
python3 -m pytest tests/ -v --tb=short
```

Expected: All tests passing, no regressions

**Step 4: Check code coverage**

```bash
python3 -m pytest tests/test_batch_embedding_generator.py --cov=codesearch.embeddings.batch_generator --cov-report=term-missing
```

Expected: >90% coverage on batch_generator.py

**Step 5: Code formatting**

```bash
python3 -m ruff check codesearch/embeddings/batch_generator.py --fix
```

**Step 6: Commit final test**

```bash
git add tests/test_batch_embedding_generator.py
git commit -m "feat: Add real codebase validation test (Task 7)"
```

**Step 7: Create PR**

```bash
gh pr create \
  --title "feat: Implement batch embedding generation with caching (Issue #7)" \
  --body "Implement BatchEmbeddingGenerator for efficient batch processing of code embeddings with persistent caching. Features:

- Sequential batch processing of functions and classes
- In-memory cache + disk persistence for embeddings
- Graceful error handling (failed items don't stop batch)
- Comprehensive error tracking and summary statistics
- Real-world validation on P3 codebase (100+ functions)
- 90%+ code coverage with 20+ tests

Ready for review and merge."
```

**Step 8: Verify PR creation**

```bash
git log --oneline -1
gh pr view
```

---

## Summary

This plan implements Issue #7 (Batch Embedding Generation) in 7 focused tasks:

1. **Task 1:** Basic structure (class, methods, initialization)
2. **Task 2:** Cache system (load/save from disk)
3. **Task 3:** Core batch processing (sequential pipeline)
4. **Task 4:** Cache hit tracking (reuse existing embeddings)
5. **Task 5:** Error handling (graceful failures)
6. **Task 6:** Integration testing (with EmbeddingGenerator/TextPreparator)
7. **Task 7:** Real codebase validation and PR

**Completion Criteria:**
✓ BatchEmbeddingGenerator class fully functional
✓ Caching system working (in-memory + disk)
✓ 20+ comprehensive tests passing
✓ >90% code coverage
✓ Real codebase validation (100+ functions)
✓ PR created and ready for review

**Next Component:** Issue #8 - Embedding Quality Assurance
