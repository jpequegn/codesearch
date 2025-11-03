# Issue #5: Embedding Model Selection & Setup - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement embedding model selection and setup using CodeBERT from HuggingFace with local caching and batch processing support.

**Architecture:** Create EmbeddingModel configuration dataclass and EmbeddingGenerator class that loads/caches models from HuggingFace, generates embeddings using transformers library, and supports batch processing for efficiency.

**Tech Stack:**
- HuggingFace transformers (model loading, tokenization)
- PyTorch (inference, device management)
- CodeBERT (microsoft/codebert-base)

---

## Task 1: Create EmbeddingModel Configuration Dataclass

**Files:**
- Modify: `codesearch/models.py` (add new dataclass at end)
- Test: `tests/test_embedding_generator.py` (create new file)

**Step 1: Write failing test**

Create `tests/test_embedding_generator.py` with this content:

```python
"""Tests for embedding model and generator."""

import pytest
from codesearch.models import EmbeddingModel


def test_embedding_model_creation():
    """Test creating an EmbeddingModel configuration."""
    model = EmbeddingModel(
        name="codebert-base",
        model_name="microsoft/codebert-base",
        dimensions=768,
        max_length=512,
    )

    assert model.name == "codebert-base"
    assert model.model_name == "microsoft/codebert-base"
    assert model.dimensions == 768
    assert model.max_length == 512
    assert model.device == "auto"  # Default value


def test_embedding_model_custom_device():
    """Test creating an EmbeddingModel with custom device."""
    model = EmbeddingModel(
        name="codebert-base",
        model_name="microsoft/codebert-base",
        dimensions=768,
        max_length=512,
        device="cpu",
    )

    assert model.device == "cpu"
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/julienpequegnot/Code/codesearch/.worktrees/issue-5-embedding-models
source venv/bin/activate
python -m pytest tests/test_embedding_generator.py::test_embedding_model_creation -v
```

Expected output: `FAILED ... NameError: name 'EmbeddingModel' is not defined`

**Step 3: Write minimal implementation**

Add to `codesearch/models.py` (at end of file, before last line):

```python
@dataclass
class EmbeddingModel:
    """Configuration for an embedding model."""

    name: str                      # e.g., "codebert-base"
    model_name: str                # HuggingFace model ID (e.g., "microsoft/codebert-base")
    dimensions: int                # Vector size (e.g., 768)
    max_length: int                # Max input tokens (e.g., 512)
    device: str = "auto"           # "cpu", "cuda", or "auto"
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_embedding_generator.py::test_embedding_model_creation tests/test_embedding_generator.py::test_embedding_model_custom_device -v
```

Expected: Both tests PASS

**Step 5: Commit**

```bash
git add codesearch/models.py tests/test_embedding_generator.py
git commit -m "feat: Add EmbeddingModel configuration dataclass"
```

---

## Task 2: Create EmbeddingGenerator Basic Structure

**Files:**
- Create: `codesearch/embeddings/generator.py`
- Modify: `codesearch/embeddings/__init__.py`
- Modify: `tests/test_embedding_generator.py` (add tests)

**Step 1: Write failing test for initialization**

Add to `tests/test_embedding_generator.py`:

```python
from codesearch.embeddings.generator import EmbeddingGenerator


def test_embedding_generator_initialization():
    """Test initializing EmbeddingGenerator with default CodeBERT model."""
    generator = EmbeddingGenerator()

    assert generator is not None
    assert generator.model is not None
    assert generator.tokenizer is not None
    assert generator.device is not None


def test_embedding_generator_with_custom_model():
    """Test initializing EmbeddingGenerator with custom model config."""
    model_config = EmbeddingModel(
        name="codebert-base",
        model_name="microsoft/codebert-base",
        dimensions=768,
        max_length=512,
        device="cpu",
    )

    generator = EmbeddingGenerator(model_config)
    assert generator.model_config == model_config
    assert generator.device == "cpu"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_embedding_generator.py::test_embedding_generator_initialization -v
```

Expected: `FAILED ... ModuleNotFoundError: No module named 'codesearch.embeddings.generator'`

**Step 3: Create generator module with minimal implementation**

Create `codesearch/embeddings/generator.py`:

```python
"""Embedding generation using transformer models."""

from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
import torch

from codesearch.models import EmbeddingModel


class EmbeddingGenerator:
    """Generates embeddings for code using HuggingFace models."""

    # Default CodeBERT model configuration
    DEFAULT_MODEL = EmbeddingModel(
        name="codebert-base",
        model_name="microsoft/codebert-base",
        dimensions=768,
        max_length=512,
        device="auto",
    )

    def __init__(self, model_config: Optional[EmbeddingModel] = None):
        """
        Initialize the embedding generator.

        Args:
            model_config: EmbeddingModel configuration (uses default CodeBERT if None)
        """
        self.model_config = model_config or self.DEFAULT_MODEL

        # Determine device
        if self.model_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.model_config.device

        # Load model and tokenizer from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name
        )
        self.model = AutoModel.from_pretrained(
            self.model_config.model_name
        ).to(self.device)

        # Set to eval mode (no gradients)
        self.model.eval()

    def embed_code(self, code_text: str) -> List[float]:
        """
        Generate embedding for a single code snippet.

        Args:
            code_text: Source code as string

        Returns:
            768-dimensional embedding vector (normalized)
        """
        # Tokenize input
        inputs = self.tokenizer(
            code_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_length,
            padding=True,
        ).to(self.device)

        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu()

        # L2 normalize
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding[0].tolist()

    def get_model_info(self) -> dict:
        """Return model metadata."""
        return {
            "name": self.model_config.name,
            "model_name": self.model_config.model_name,
            "dimensions": self.model_config.dimensions,
            "max_length": self.model_config.max_length,
            "device": self.device,
        }
```

**Step 4: Update embeddings __init__.py**

Modify `codesearch/embeddings/__init__.py`:

```python
"""Embedding generation module."""

from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.models import EmbeddingModel

__all__ = ["EmbeddingGenerator", "EmbeddingModel"]
```

**Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_embedding_generator.py::test_embedding_generator_initialization tests/test_embedding_generator.py::test_embedding_generator_with_custom_model -v
```

Expected: Both tests PASS

**Step 6: Commit**

```bash
git add codesearch/embeddings/generator.py codesearch/embeddings/__init__.py tests/test_embedding_generator.py
git commit -m "feat: Implement EmbeddingGenerator with CodeBERT model loading"
```

---

## Task 3: Implement Single Embedding Generation & Testing

**Files:**
- Modify: `tests/test_embedding_generator.py` (add tests)

**Step 1: Write failing tests for embed_code**

Add to `tests/test_embedding_generator.py`:

```python
def test_embed_code_simple():
    """Test embedding a simple code snippet."""
    generator = EmbeddingGenerator()

    code = "def hello():\n    return 'world'"
    embedding = generator.embed_code(code)

    # Should return 768-dimensional vector
    assert isinstance(embedding, list)
    assert len(embedding) == 768
    # Values should be floats in roughly [-1, 1] range after normalization
    assert all(isinstance(x, float) for x in embedding)


def test_embed_code_consistency():
    """Test that embedding same code twice gives same result."""
    generator = EmbeddingGenerator()

    code = "def add(a, b):\n    return a + b"
    embedding1 = generator.embed_code(code)
    embedding2 = generator.embed_code(code)

    # Should be identical (deterministic)
    assert embedding1 == embedding2


def test_embed_code_different_inputs():
    """Test that different code produces different embeddings."""
    generator = EmbeddingGenerator()

    code1 = "def add(a, b):\n    return a + b"
    code2 = "def multiply(a, b):\n    return a * b"

    embedding1 = generator.embed_code(code1)
    embedding2 = generator.embed_code(code2)

    # Should be different
    assert embedding1 != embedding2


def test_embed_code_empty():
    """Test embedding empty code string."""
    generator = EmbeddingGenerator()

    embedding = generator.embed_code("")

    # Should still return valid embedding
    assert isinstance(embedding, list)
    assert len(embedding) == 768


def test_embed_code_long_truncation():
    """Test that very long code gets truncated to max_length."""
    generator = EmbeddingGenerator()

    # Create code longer than max_length tokens
    long_code = "def func():\n    " + "x = 1\n    " * 500

    # Should not raise error, should truncate
    embedding = generator.embed_code(long_code)
    assert isinstance(embedding, list)
    assert len(embedding) == 768


def test_get_model_info():
    """Test retrieving model metadata."""
    generator = EmbeddingGenerator()
    info = generator.get_model_info()

    assert info["name"] == "codebert-base"
    assert info["model_name"] == "microsoft/codebert-base"
    assert info["dimensions"] == 768
    assert info["max_length"] == 512
    assert info["device"] in ["cuda", "cpu"]
```

**Step 2: Run tests to verify they fail initially**

```bash
python -m pytest tests/test_embedding_generator.py::test_embed_code_simple -v
```

Expected: Tests PASS (implementation already handles this)

**Step 3: Run all tests together**

```bash
python -m pytest tests/test_embedding_generator.py -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_embedding_generator.py
git commit -m "test: Add comprehensive tests for single embedding generation"
```

---

## Task 4: Implement Batch Embedding Generation

**Files:**
- Modify: `codesearch/embeddings/generator.py` (add method)
- Modify: `tests/test_embedding_generator.py` (add tests)

**Step 1: Write failing tests for embed_batch**

Add to `tests/test_embedding_generator.py`:

```python
def test_embed_batch_basic():
    """Test batch embedding generation."""
    generator = EmbeddingGenerator()

    codes = [
        "def add(a, b):\n    return a + b",
        "def subtract(a, b):\n    return a - b",
        "def multiply(a, b):\n    return a * b",
    ]

    embeddings = generator.embed_batch(codes)

    # Should return list of embeddings
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    # Each embedding should be 768-dimensional
    for emb in embeddings:
        assert isinstance(emb, list)
        assert len(emb) == 768


def test_embed_batch_single_item():
    """Test batch embedding with single item."""
    generator = EmbeddingGenerator()

    codes = ["def foo(): pass"]
    embeddings = generator.embed_batch(codes)

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768


def test_embed_batch_empty_list():
    """Test batch embedding with empty list."""
    generator = EmbeddingGenerator()

    embeddings = generator.embed_batch([])

    assert embeddings == []


def test_embed_batch_consistency_with_single():
    """Test that batch results match single embedding results."""
    generator = EmbeddingGenerator()

    code = "def hello(): return 'world'"

    # Get single embedding
    single_emb = generator.embed_code(code)

    # Get batch embedding
    batch_embs = generator.embed_batch([code])

    # Should match
    assert single_emb == batch_embs[0]
```

**Step 2: Implement embed_batch method**

Add to `codesearch/embeddings/generator.py` (after embed_code method):

```python
    def embed_batch(self, code_texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple code snippets (batch).

        Args:
            code_texts: List of code snippets

        Returns:
            List of embedding vectors (each is 768-dimensional, normalized)
        """
        if not code_texts:
            return []

        # Tokenize all inputs at once
        inputs = self.tokenizer(
            code_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_length,
            padding=True,
        ).to(self.device)

        # Generate embeddings for all at once
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)

        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list of lists and move to CPU
        return embeddings.cpu().tolist()
```

**Step 3: Run tests to verify they pass**

```bash
python -m pytest tests/test_embedding_generator.py::test_embed_batch_basic tests/test_embedding_generator.py::test_embed_batch_consistency_with_single -v
```

Expected: All batch tests PASS

**Step 4: Run all tests**

```bash
python -m pytest tests/test_embedding_generator.py -v
```

Expected: All tests PASS (10+ tests)

**Step 5: Commit**

```bash
git add codesearch/embeddings/generator.py tests/test_embedding_generator.py
git commit -m "feat: Implement batch embedding generation for efficiency"
```

---

## Task 5: Test on Real Codebase & Validate Performance

**Files:**
- Test: Real P3 codebase validation

**Step 1: Create validation script**

Create temporary validation script to test on P3 codebase:

```bash
cd /Users/julienpequegnot/Code/codesearch/.worktrees/issue-5-embedding-models
source venv/bin/activate

python3 << 'EOF'
import sys
sys.path.insert(0, '.')

from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.parsers.python_parser import PythonParser
from codesearch.indexing.scanner import RepositoryScannerImpl

# Initialize components
scanner = RepositoryScannerImpl()
parser = PythonParser()
generator = EmbeddingGenerator()

print("Testing embedding generation on P3 codebase...")
print("-" * 60)

# Scan P3 repository
p3_path = "/Users/julienpequegnot/Code/parakeet-podcast-processor/p3"
files = scanner.scan_repository(p3_path, "p3")
print(f"✓ Scanned {len(files)} Python files")

# Parse first file and generate embeddings
if files:
    first_file = files[0].file_path
    print(f"\n✓ Parsing {first_file.split('/')[-1]}...")

    extracted = parser.parse_file(first_file)
    functions = [item for item in extracted if hasattr(item, 'signature')]

    if functions:
        print(f"  Found {len(functions)} functions/methods")

        # Generate embeddings for first 3 functions
        test_functions = functions[:3]
        code_snippets = [f.source_code for f in test_functions]

        print(f"\n✓ Generating embeddings for {len(test_functions)} functions...")
        embeddings = generator.embed_batch(code_snippets)

        print(f"✓ Generated {len(embeddings)} embeddings")
        for i, (func, emb) in enumerate(zip(test_functions, embeddings)):
            print(f"  [{i+1}] {func.name}: {len(emb)}-dim vector (sample: {emb[:3]})")

        # Test single embedding
        print(f"\n✓ Testing single embedding generation...")
        single_emb = generator.embed_code(code_snippets[0])
        print(f"  Single embedding: {len(single_emb)}-dim (matches batch: {single_emb == embeddings[0]})")

        # Get model info
        print(f"\n✓ Model info:")
        info = generator.get_model_info()
        for key, val in info.items():
            print(f"  {key}: {val}")

        print("\n✅ All validation tests passed!")
    else:
        print("  No functions found in parsed file")
else:
    print("No Python files found")
EOF
```

**Step 2: Run validation**

Expected output:
```
✓ Scanned 13 Python files
✓ Parsing cli.py...
  Found X functions/methods
✓ Generating embeddings for 3 functions...
✓ Generated 3 embeddings
  [1] function_name: 768-dim vector
  ...
✓ Testing single embedding generation...
  Single embedding: 768-dim (matches batch: True)
✓ Model info:
  name: codebert-base
  ...
✅ All validation tests passed!
```

**Step 3: Check test coverage**

```bash
python -m pytest tests/test_embedding_generator.py -v --cov=codesearch.embeddings --cov-report=term-missing
```

Expected: >90% coverage on generator.py

**Step 4: Verify all project tests still pass**

```bash
python -m pytest tests/ -v
```

Expected: 61+ tests passing (original tests + new embedding tests)

**Step 5: Commit**

```bash
git add tests/test_embedding_generator.py
git commit -m "test: Validate embedding generation on real P3 codebase"
```

---

## Task 6: Create Configuration File (Optional Enhancement)

**Files:**
- Create: `config/embedding_models.yaml`
- Modify: `codesearch/embeddings/generator.py` (add config loading)

**Step 1: Create embedding models configuration**

Create `config/embedding_models.yaml`:

```yaml
# Embedding model configurations
models:
  codebert-base:
    name: codebert-base
    model_name: microsoft/codebert-base
    dimensions: 768
    max_length: 512
    description: "CodeBERT base model, balanced performance"
    languages: [python, javascript, java, go]

  codebert-small:
    name: codebert-small
    model_name: microsoft/codebert-small
    dimensions: 384
    max_length: 512
    description: "Smaller, faster CodeBERT variant"
    languages: [python, javascript, java, go]

  codet5-base:
    name: codet5-base
    model_name: Salesforce/codet5-base
    dimensions: 768
    max_length: 512
    description: "CodeT5 model with better semantic understanding"
    languages: [python, javascript, java, go, php, ruby]

# Default model to use
default: codebert-base

# Cache directory for downloaded models
cache_dir: ~/.codesearch/models

# Device preference
device: auto  # auto (cuda if available, else cpu), cuda, cpu
```

**Step 2: Add test for config loading**

Add to `tests/test_embedding_generator.py`:

```python
import yaml
import os


def test_embedding_models_config_exists():
    """Test that embedding models config file exists."""
    config_path = "config/embedding_models.yaml"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"


def test_embedding_models_config_valid():
    """Test that config file is valid YAML with required fields."""
    with open("config/embedding_models.yaml", "r") as f:
        config = yaml.safe_load(f)

    assert "models" in config
    assert "default" in config
    assert config["default"] in config["models"]

    # Each model should have required fields
    for model_name, model_config in config["models"].items():
        assert "model_name" in model_config
        assert "dimensions" in model_config
        assert "max_length" in model_config
```

**Step 3: Run config validation tests**

```bash
python -m pytest tests/test_embedding_generator.py::test_embedding_models_config_exists -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add config/embedding_models.yaml tests/test_embedding_generator.py
git commit -m "config: Add embedding models configuration file"
```

---

## Task 7: Final Testing & PR Creation

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -v --cov=codesearch --cov-report=term-missing
```

Expected:
- All tests PASS
- Coverage >85% on codesearch/embeddings/generator.py
- Total coverage >88%

**Step 2: Run linting checks**

```bash
python -m black --check codesearch/embeddings/
python -m ruff check codesearch/embeddings/
```

Expected: No issues (or auto-fix with `black codesearch/embeddings/`)

**Step 3: Check Git status**

```bash
git status
git log --oneline -7
```

Expected: All commits are present, clean working directory

**Step 4: Push feature branch**

```bash
git push origin feature/issue-5-embedding-models
```

**Step 5: Create PR**

```bash
gh pr create --title "feat: Component 2.1 - Embedding Model Selection & Setup" \
  --body "
## Summary

Implements embedding model selection and integration using CodeBERT from HuggingFace:

- EmbeddingModel dataclass for model configuration
- EmbeddingGenerator for loading models and generating embeddings
- Support for single and batch embedding generation
- L2 normalized vectors for cosine similarity in LanceDB
- Automatic device detection (GPU/CPU)
- Comprehensive testing on real P3 codebase

## Test Results

- 15+ new tests for embedding generation
- >90% coverage on generator.py
- Validated on P3 codebase (13 Python files)
- All 61+ tests passing

Fixes #5
"
```

Expected: PR created successfully

**Step 6: Final commit message**

```bash
git log --oneline -1
```

Expected output similar to:
```
feat: Implement batch embedding generation for efficiency
```

---

## Success Criteria

✅ EmbeddingModel dataclass created with all required fields
✅ EmbeddingGenerator loads CodeBERT model successfully
✅ Single embedding generation works (768-dim vectors)
✅ Batch embedding generation works and is faster
✅ L2 normalization applied to all embeddings
✅ Device auto-detection works (GPU/CPU)
✅ All 15+ tests passing with >90% coverage
✅ Validated on real P3 codebase
✅ Configuration file created with multiple models
✅ Code follows project conventions (tested with black/ruff)
✅ PR created for code review

---

## File Structure After Completion

```
codesearch/
├── embeddings/
│   ├── __init__.py (updated)
│   └── generator.py (NEW)
├── models.py (updated with EmbeddingModel)
└── ...

tests/
└── test_embedding_generator.py (NEW, 15+ tests)

config/
└── embedding_models.yaml (NEW)
```

---

## Next Steps

After PR is merged:
- Issue #5 closed
- Ready for Issue #2.2 (Text Preparation for Embeddings)
- Ready for Issue #2.3 (Batch Processing Pipeline)
- Ready for Issue #3.1 (LanceDB Schema Design)
