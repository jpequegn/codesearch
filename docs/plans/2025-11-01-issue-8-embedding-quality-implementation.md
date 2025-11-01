# Embedding Quality Assurance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Create an extensible validation system for code embeddings that tests vector properties, semantic correctness, and deterministic output.

**Architecture:** Composable validation check pipeline with a base ValidationCheck interface, three concrete implementations (VectorCheck, SimilarityCheck, ConsistencyCheck), and an EmbeddingValidator orchestrator that runs all checks and aggregates results.

**Tech Stack:** Python dataclasses, abstract base classes, cosine similarity calculations, pytest for unit testing.

---

## Task 1: ValidationCheck Interface and VectorCheck

**Files:**
- Create: `codesearch/embeddings/validator.py`
- Create: `tests/test_embedding_validator.py`
- Modify: `codesearch/embeddings/__init__.py`

**Step 1: Write the failing test for ValidationCheck interface**

Create `tests/test_embedding_validator.py`:

```python
"""Tests for embedding quality validation."""

import pytest
import math
from codesearch.embeddings.validator import (
    ValidationCheck,
    ValidationResult,
    VectorCheck,
    EmbeddingValidator
)


def test_validation_check_is_abstract():
    """ValidationCheck cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ValidationCheck()


def test_validation_result_creation():
    """ValidationResult can be created with passed flag and message."""
    result = ValidationResult(passed=True, message="All good")
    assert result.passed is True
    assert result.message == "All good"


def test_vector_check_valid_embedding():
    """VectorCheck accepts valid 768-dimensional embedding."""
    check = VectorCheck()

    # Create valid embedding: 768 floats in range [-1, 1]
    valid_embedding = [0.5] * 768

    result = check.validate(valid_embedding)

    assert result.passed is True
    assert result.message == "Vector format valid"


def test_vector_check_wrong_dimensions():
    """VectorCheck rejects embedding with wrong dimensions."""
    check = VectorCheck()

    # Wrong dimensions: 512 instead of 768
    wrong_dims = [0.5] * 512

    result = check.validate(wrong_dims)

    assert result.passed is False
    assert "Expected 768 dims" in result.message


def test_vector_check_detects_nan():
    """VectorCheck detects NaN values in embedding."""
    check = VectorCheck()

    # Valid embedding with NaN injected
    embedding = [0.5] * 768
    embedding[100] = float('nan')

    result = check.validate(embedding)

    assert result.passed is False
    assert "Invalid value" in result.message


def test_vector_check_detects_infinity():
    """VectorCheck detects infinity values in embedding."""
    check = VectorCheck()

    # Valid embedding with infinity injected
    embedding = [0.5] * 768
    embedding[200] = float('inf')

    result = check.validate(embedding)

    assert result.passed is False
    assert "Invalid value" in result.message


def test_vector_check_out_of_range():
    """VectorCheck rejects embeddings with out-of-range values."""
    check = VectorCheck()

    # Embedding with extreme values
    embedding = [3.0] * 768  # Greater than allowed [-2, 2]

    result = check.validate(embedding)

    assert result.passed is False
    assert "Out of range" in result.message


def test_vector_check_boundary_valid():
    """VectorCheck accepts values at boundaries [-2, 2]."""
    check = VectorCheck()

    # Embedding at the extremes but valid
    embedding = [2.0] * 384 + [-2.0] * 384

    result = check.validate(embedding)

    assert result.passed is True


def test_vector_check_name_property():
    """VectorCheck has a name property."""
    check = VectorCheck()

    assert check.name == "vector_format"


def test_vector_check_mixed_types():
    """VectorCheck rejects non-float values in embedding."""
    check = VectorCheck()

    # Valid dimensions but contains non-float
    embedding = [0.5] * 767 + ["invalid"]  # type: ignore

    result = check.validate(embedding)

    assert result.passed is False
    assert "not float" in result.message
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_embedding_validator.py::test_validation_check_is_abstract -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'codesearch.embeddings.validator'"

**Step 3: Write minimal ValidationCheck interface and VectorCheck**

Create `codesearch/embeddings/validator.py`:

```python
"""Embedding quality validation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
import math


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    passed: bool
    message: str


class ValidationCheck(ABC):
    """Base class for all embedding validation checks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this check."""
        pass

    @abstractmethod
    def validate(self, embedding: List[float]) -> ValidationResult:
        """Run validation on an embedding.

        Args:
            embedding: 768-dimensional embedding vector

        Returns:
            ValidationResult with passed flag and message
        """
        pass


class VectorCheck(ValidationCheck):
    """Validates embedding vector format and properties."""

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

    @property
    def name(self) -> str:
        return "vector_format"

    def validate(self, embedding: List[float]) -> ValidationResult:
        """Check dimensions, detect NaN/infinity, validate value ranges."""
        # Check 1: Dimensions must be 768 for CodeBERT
        if len(embedding) != self.dimensions:
            return ValidationResult(
                passed=False,
                message=f"Expected {self.dimensions} dims, got {len(embedding)}"
            )

        # Check 2: Detect NaN/Infinity and type validation
        for i, val in enumerate(embedding):
            if not isinstance(val, float):
                return ValidationResult(
                    passed=False,
                    message=f"Element {i} not float: {type(val)}"
                )
            if math.isnan(val) or math.isinf(val):
                return ValidationResult(
                    passed=False,
                    message=f"Invalid value at index {i}: {val}"
                )

        # Check 3: Value range (after normalization, typically [-1, 1])
        min_val = min(embedding)
        max_val = max(embedding)
        if not (-2.0 <= min_val <= 2.0 and -2.0 <= max_val <= 2.0):
            return ValidationResult(
                passed=False,
                message=f"Out of range: min={min_val:.3f}, max={max_val:.3f}"
            )

        return ValidationResult(passed=True, message="Vector format valid")
```

**Step 4: Run all tests to verify they pass**

Run: `python3 -m pytest tests/test_embedding_validator.py::test_validation_check_is_abstract tests/test_embedding_validator.py::test_validation_result_creation tests/test_embedding_validator.py::test_vector_check_valid_embedding tests/test_embedding_validator.py::test_vector_check_wrong_dimensions tests/test_embedding_validator.py::test_vector_check_detects_nan tests/test_embedding_validator.py::test_vector_check_detects_infinity tests/test_embedding_validator.py::test_vector_check_out_of_range tests/test_embedding_validator.py::test_vector_check_boundary_valid tests/test_embedding_validator.py::test_vector_check_name_property tests/test_embedding_validator.py::test_vector_check_mixed_types -v`

Expected: 10 PASSED

**Step 5: Update embeddings __init__.py**

Modify `codesearch/embeddings/__init__.py`:

```python
"""Embedding generation and validation module."""

from codesearch.embeddings.text_preparator import TextPreparator
from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.embeddings.batch_generator import BatchEmbeddingGenerator
from codesearch.embeddings.validator import (
    ValidationCheck,
    ValidationResult,
    VectorCheck,
    EmbeddingValidator
)
from codesearch.models import EmbeddingModel

__all__ = [
    "TextPreparator",
    "EmbeddingGenerator",
    "BatchEmbeddingGenerator",
    "ValidationCheck",
    "ValidationResult",
    "VectorCheck",
    "EmbeddingValidator",
    "EmbeddingModel"
]
```

**Step 6: Commit**

```bash
git add codesearch/embeddings/validator.py tests/test_embedding_validator.py codesearch/embeddings/__init__.py
git commit -m "feat: Create ValidationCheck interface and VectorCheck implementation (Task 1)"
```

---

## Task 2: SimilarityCheck with Pattern Pairs

**Files:**
- Modify: `codesearch/embeddings/validator.py` (add SimilarityCheck and TEST_PATTERNS)
- Modify: `tests/test_embedding_validator.py` (add similarity tests)

**Step 1: Write the failing test for SimilarityCheck**

Add to `tests/test_embedding_validator.py`:

```python
from codesearch.embeddings.validator import SimilarityCheck
from codesearch.embeddings.generator import EmbeddingGenerator


def test_similarity_check_similar_patterns():
    """SimilarityCheck detects similar code patterns."""
    generator = EmbeddingGenerator()
    check = SimilarityCheck(generator)

    # Create a valid embedding (not used directly by SimilarityCheck, but required)
    dummy_embedding = [0.5] * 768

    result = check.validate(dummy_embedding)

    # Should pass because similar patterns ARE similar
    assert result.passed is True


def test_similarity_check_name_property():
    """SimilarityCheck has correct name property."""
    generator = EmbeddingGenerator()
    check = SimilarityCheck(generator)

    assert check.name == "semantic_correctness"


def test_similarity_check_with_custom_tolerance():
    """SimilarityCheck respects tolerance parameter."""
    generator = EmbeddingGenerator()

    # Tight tolerance
    check_tight = SimilarityCheck(generator, tolerance=0.01)
    assert check_tight.tolerance == 0.01

    # Loose tolerance
    check_loose = SimilarityCheck(generator, tolerance=0.20)
    assert check_loose.tolerance == 0.20


def test_similarity_check_cosine_similarity_calculation():
    """SimilarityCheck calculates cosine similarity correctly."""
    generator = EmbeddingGenerator()
    check = SimilarityCheck(generator)

    # Test vectors: identical
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    similarity = check._cosine_similarity(v1, v2)
    assert abs(similarity - 1.0) < 0.001  # Should be 1.0

    # Test vectors: orthogonal (perpendicular)
    v3 = [1.0, 0.0, 0.0]
    v4 = [0.0, 1.0, 0.0]
    similarity = check._cosine_similarity(v3, v4)
    assert abs(similarity - 0.0) < 0.001  # Should be 0.0


def test_similarity_check_detects_unrelated_patterns():
    """SimilarityCheck can detect when unrelated patterns are too similar."""
    generator = EmbeddingGenerator()

    # Hypothetical: if embeddings were identical for unrelated code
    # This would fail the check
    check = SimilarityCheck(generator)
    dummy_embedding = [0.5] * 768

    # In practice, this should pass because CodeBERT correctly
    # separates unrelated code patterns
    result = check.validate(dummy_embedding)
    assert result.passed is True
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_embedding_validator.py::test_similarity_check_similar_patterns -v`

Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Implement SimilarityCheck**

Modify `codesearch/embeddings/validator.py` - add after VectorCheck:

```python
# Test patterns for semantic correctness validation
TEST_PATTERNS = {
    "similar_arithmetic": {
        "code1": "def add(a, b):\n    return a + b",
        "code2": "def subtract(a, b):\n    return a - b",
        "expected_similarity": (0.75, 1.0),
        "reason": "Both are basic arithmetic operations"
    },
    "similar_logic": {
        "code1": "if x > 0:\n    return True",
        "code2": "if x >= 1:\n    return True",
        "expected_similarity": (0.70, 1.0),
        "reason": "Both implement conditional logic"
    },
    "unrelated_arithmetic_parse": {
        "code1": "def add(a, b):\n    return a + b",
        "code2": "def parse_json(text):\n    return json.loads(text)",
        "expected_similarity": (0.0, 0.50),
        "reason": "Different domains: math vs parsing"
    },
    "unrelated_math_network": {
        "code1": "def factorial(n):\n    return n * (n-1)",
        "code2": "def http_request(url):\n    return requests.get(url)",
        "expected_similarity": (0.0, 0.40),
        "reason": "Different domains: math vs networking"
    }
}


class SimilarityCheck(ValidationCheck):
    """Validates semantic correctness via known code pattern pairs."""

    def __init__(
        self,
        embedding_generator,
        patterns: Dict = None,
        tolerance: float = 0.05
    ):
        self.generator = embedding_generator
        self.patterns = patterns or TEST_PATTERNS
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return "semantic_correctness"

    def validate(self, embedding: List[float]) -> ValidationResult:
        """Test known pattern relationships.

        For each pattern pair, verify cosine similarity falls within expected bounds.
        """
        failed_patterns = []

        for pattern_name, pattern_data in self.patterns.items():
            code1 = pattern_data["code1"]
            code2 = pattern_data["code2"]
            expected_min, expected_max = pattern_data["expected_similarity"]

            # Generate embeddings
            emb1 = self.generator.embed_code(code1)
            emb2 = self.generator.embed_code(code2)

            # Calculate cosine similarity
            similarity = self._cosine_similarity(emb1, emb2)

            # Check if within bounds (with tolerance)
            lower_bound = expected_min - self.tolerance
            upper_bound = expected_max + self.tolerance

            if not (lower_bound <= similarity <= upper_bound):
                failed_patterns.append({
                    "pattern": pattern_name,
                    "expected": (expected_min, expected_max),
                    "actual": similarity,
                    "reason": pattern_data["reason"]
                })

        if failed_patterns:
            messages = [
                f"{p['pattern']}: expected {p['expected']}, got {p['actual']:.3f} ({p['reason']})"
                for p in failed_patterns
            ]
            return ValidationResult(
                passed=False,
                message=f"Semantic checks failed: {'; '.join(messages)}"
            )

        return ValidationResult(passed=True, message="Semantic similarity correct")

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(b * b for b in v2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_embedding_validator.py::test_similarity_check_similar_patterns tests/test_embedding_validator.py::test_similarity_check_name_property tests/test_embedding_validator.py::test_similarity_check_with_custom_tolerance tests/test_embedding_validator.py::test_similarity_check_cosine_similarity_calculation tests/test_embedding_validator.py::test_similarity_check_detects_unrelated_patterns -v`

Expected: 5 PASSED

**Step 5: Commit**

```bash
git add codesearch/embeddings/validator.py tests/test_embedding_validator.py
git commit -m "feat: Implement SimilarityCheck with test patterns (Task 2)"
```

---

## Task 3: ConsistencyCheck

**Files:**
- Modify: `codesearch/embeddings/validator.py` (add ConsistencyCheck)
- Modify: `tests/test_embedding_validator.py` (add consistency tests)

**Step 1: Write the failing test for ConsistencyCheck**

Add to `tests/test_embedding_validator.py`:

```python
from codesearch.embeddings.validator import ConsistencyCheck


def test_consistency_check_deterministic():
    """ConsistencyCheck validates deterministic embedding output."""
    generator = EmbeddingGenerator()
    check = ConsistencyCheck(generator, runs=3)

    dummy_embedding = [0.5] * 768
    result = check.validate(dummy_embedding)

    # Should pass because CodeBERT is deterministic
    assert result.passed is True


def test_consistency_check_name_property():
    """ConsistencyCheck has correct name property."""
    generator = EmbeddingGenerator()
    check = ConsistencyCheck(generator)

    assert check.name == "deterministic_output"


def test_consistency_check_configurable_runs():
    """ConsistencyCheck respects runs parameter."""
    generator = EmbeddingGenerator()

    check_3_runs = ConsistencyCheck(generator, runs=3)
    assert check_3_runs.runs == 3

    check_5_runs = ConsistencyCheck(generator, runs=5)
    assert check_5_runs.runs == 5
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_embedding_validator.py::test_consistency_check_deterministic -v`

Expected: FAIL with "ImportError: cannot import name 'ConsistencyCheck'"

**Step 3: Implement ConsistencyCheck**

Modify `codesearch/embeddings/validator.py` - add after SimilarityCheck:

```python
class ConsistencyCheck(ValidationCheck):
    """Validates that embeddings are deterministic across runs."""

    def __init__(self, embedding_generator, runs: int = 3):
        self.generator = embedding_generator
        self.runs = runs

    @property
    def name(self) -> str:
        return "deterministic_output"

    def validate(self, embedding: List[float]) -> ValidationResult:
        """Test determinism by re-embedding test code multiple times."""
        test_code = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""

        # Embed same code multiple times
        embeddings = [self.generator.embed_code(test_code) for _ in range(self.runs)]

        # All embeddings must be identical
        first_embedding = embeddings[0]

        for i, emb in enumerate(embeddings[1:], 1):
            if emb != first_embedding:
                # Check how different they are
                diff_count = sum(1 for a, b in zip(first_embedding, emb) if a != b)
                return ValidationResult(
                    passed=False,
                    message=f"Run {i} differs: {diff_count}/{len(emb)} elements changed"
                )

        return ValidationResult(passed=True, message="Output deterministic")
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_embedding_validator.py::test_consistency_check_deterministic tests/test_embedding_validator.py::test_consistency_check_name_property tests/test_embedding_validator.py::test_consistency_check_configurable_runs -v`

Expected: 3 PASSED

**Step 5: Commit**

```bash
git add codesearch/embeddings/validator.py tests/test_embedding_validator.py
git commit -m "feat: Implement ConsistencyCheck for determinism validation (Task 3)"
```

---

## Task 4: EmbeddingValidator Orchestrator

**Files:**
- Modify: `codesearch/embeddings/validator.py` (add ValidationReport and EmbeddingValidator)
- Modify: `tests/test_embedding_validator.py` (add orchestrator tests)

**Step 1: Write the failing tests for EmbeddingValidator**

Add to `tests/test_embedding_validator.py`:

```python
from codesearch.embeddings.validator import ValidationReport, EmbeddingValidator


def test_embedding_validator_initialization():
    """EmbeddingValidator starts with no checks registered."""
    validator = EmbeddingValidator()

    assert len(validator.checks) == 0


def test_embedding_validator_register_check():
    """EmbeddingValidator can register checks."""
    validator = EmbeddingValidator()
    check = VectorCheck()

    validator.register_check(check)

    assert len(validator.checks) == 1
    assert validator.checks[0] == check


def test_embedding_validator_single_check_pass():
    """EmbeddingValidator returns passing report when all checks pass."""
    validator = EmbeddingValidator()
    check = VectorCheck()
    validator.register_check(check)

    valid_embedding = [0.5] * 768
    report = validator.validate(valid_embedding)

    assert report.embedding_valid is True
    assert len(report.checks_passed) == 1
    assert len(report.checks_failed) == 0


def test_embedding_validator_single_check_fail():
    """EmbeddingValidator returns failing report when check fails."""
    validator = EmbeddingValidator()
    check = VectorCheck()
    validator.register_check(check)

    invalid_embedding = [0.5] * 512  # Wrong dimensions
    report = validator.validate(invalid_embedding)

    assert report.embedding_valid is False
    assert len(report.checks_passed) == 0
    assert len(report.checks_failed) == 1
    assert "vector_format" in report.checks_failed


def test_embedding_validator_multiple_checks_mixed():
    """EmbeddingValidator aggregates results from multiple checks."""
    generator = EmbeddingGenerator()
    validator = EmbeddingValidator()

    validator.register_check(VectorCheck())
    validator.register_check(SimilarityCheck(generator))
    validator.register_check(ConsistencyCheck(generator))

    valid_embedding = [0.5] * 768
    report = validator.validate(valid_embedding)

    # All checks should pass for valid embedding
    assert report.embedding_valid is True
    assert len(report.checks_passed) >= 1


def test_embedding_validator_validation_report():
    """ValidationReport provides human-readable summary."""
    report = ValidationReport(
        embedding_valid=True,
        timestamp="2025-11-01T12:00:00",
        checks_passed=["vector_format", "semantic_correctness"],
        checks_failed=[],
        messages={}
    )

    summary = report.summary()
    assert "All" in summary
    assert "checks passed" in summary


def test_embedding_validator_batch_validation():
    """EmbeddingValidator can validate multiple embeddings."""
    validator = EmbeddingValidator()
    validator.register_check(VectorCheck())

    embeddings = [[0.5] * 768, [0.7] * 768, [0.3] * 768]
    reports = validator.validate_batch(embeddings)

    assert len(reports) == 3
    assert all(r.embedding_valid for r in reports)


def test_embedding_validator_failed_check_message_included():
    """ValidationReport includes messages from failed checks."""
    validator = EmbeddingValidator()
    validator.register_check(VectorCheck())

    invalid_embedding = [0.5] * 512  # Wrong dimensions
    report = validator.validate(invalid_embedding)

    assert report.embedding_valid is False
    assert len(report.messages) > 0
    assert "vector_format" in report.messages
    assert "Expected 768 dims" in report.messages["vector_format"]
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_embedding_validator.py::test_embedding_validator_initialization -v`

Expected: FAIL with "ImportError: cannot import name 'EmbeddingValidator'"

**Step 3: Implement EmbeddingValidator and ValidationReport**

Modify `codesearch/embeddings/validator.py` - add after ConsistencyCheck:

```python
@dataclass
class ValidationReport:
    """Complete validation report for an embedding."""
    embedding_valid: bool  # Overall pass/fail
    timestamp: str
    checks_passed: List[str]  # Names of passing checks
    checks_failed: List[str]  # Names of failing checks
    messages: Dict[str, str]  # Detailed message per check

    def summary(self) -> str:
        """Human-readable summary."""
        if self.embedding_valid:
            return f"✓ All {len(self.checks_passed)} checks passed"
        else:
            total = len(self.checks_passed) + len(self.checks_failed)
            return f"✗ {len(self.checks_failed)} of {total} checks failed"


class EmbeddingValidator:
    """Orchestrates validation checks and aggregates results."""

    def __init__(self):
        self.checks: List[ValidationCheck] = []

    def register_check(self, check: ValidationCheck) -> None:
        """Register a validation check.

        Args:
            check: ValidationCheck instance to add
        """
        self.checks.append(check)

    def validate(self, embedding: List[float]) -> ValidationReport:
        """Run all registered checks on an embedding.

        Args:
            embedding: 768-dimensional embedding vector

        Returns:
            ValidationReport with pass/fail status and detailed results
        """
        passed_checks = []
        failed_checks = []
        messages = {}

        for check in self.checks:
            result = check.validate(embedding)
            check_name = check.name

            if result.passed:
                passed_checks.append(check_name)
            else:
                failed_checks.append(check_name)
                messages[check_name] = result.message

        return ValidationReport(
            embedding_valid=len(failed_checks) == 0,
            timestamp=datetime.utcnow().isoformat(),
            checks_passed=passed_checks,
            checks_failed=failed_checks,
            messages=messages
        )

    def validate_batch(self, embeddings: List[List[float]]) -> List[ValidationReport]:
        """Validate multiple embeddings.

        Args:
            embeddings: List of 768-dimensional vectors

        Returns:
            List of ValidationReport, one per embedding
        """
        return [self.validate(emb) for emb in embeddings]
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_embedding_validator.py::test_embedding_validator_initialization tests/test_embedding_validator.py::test_embedding_validator_register_check tests/test_embedding_validator.py::test_embedding_validator_single_check_pass tests/test_embedding_validator.py::test_embedding_validator_single_check_fail tests/test_embedding_validator.py::test_embedding_validator_multiple_checks_mixed tests/test_embedding_validator.py::test_embedding_validator_validation_report tests/test_embedding_validator.py::test_embedding_validator_batch_validation tests/test_embedding_validator.py::test_embedding_validator_failed_check_message_included -v`

Expected: 8 PASSED

**Step 5: Commit**

```bash
git add codesearch/embeddings/validator.py tests/test_embedding_validator.py
git commit -m "feat: Implement EmbeddingValidator orchestrator with batch support (Task 4)"
```

---

## Task 5: Comprehensive Unit Tests and Coverage

**Files:**
- Modify: `tests/test_embedding_validator.py` (add edge case and integration tests)

**Step 1: Add comprehensive edge case tests**

Add to `tests/test_embedding_validator.py`:

```python
def test_vector_check_empty_embedding():
    """VectorCheck rejects empty embedding."""
    check = VectorCheck()

    empty_embedding = []
    result = check.validate(empty_embedding)

    assert result.passed is False
    assert "Expected 768 dims" in result.message


def test_similarity_check_empty_patterns():
    """SimilarityCheck handles empty patterns gracefully."""
    generator = EmbeddingGenerator()
    check = SimilarityCheck(generator, patterns={})

    dummy_embedding = [0.5] * 768
    result = check.validate(dummy_embedding)

    # With no patterns to check, should pass
    assert result.passed is True


def test_consistency_check_single_run():
    """ConsistencyCheck works with single run (no comparison needed)."""
    generator = EmbeddingGenerator()
    check = ConsistencyCheck(generator, runs=1)

    dummy_embedding = [0.5] * 768
    result = check.validate(dummy_embedding)

    assert result.passed is True


def test_validator_registers_multiple_checks():
    """EmbeddingValidator can register multiple distinct checks."""
    generator = EmbeddingGenerator()
    validator = EmbeddingValidator()

    validator.register_check(VectorCheck())
    validator.register_check(SimilarityCheck(generator))
    validator.register_check(ConsistencyCheck(generator))

    assert len(validator.checks) == 3


def test_validator_check_order_preserved():
    """EmbeddingValidator runs checks in registration order."""
    validator = EmbeddingValidator()

    check1 = VectorCheck()
    check2 = SimilarityCheck(EmbeddingGenerator())

    validator.register_check(check1)
    validator.register_check(check2)

    assert validator.checks[0].name == "vector_format"
    assert validator.checks[1].name == "semantic_correctness"


def test_validation_report_timestamp_format():
    """ValidationReport timestamp is ISO format."""
    validator = EmbeddingValidator()
    validator.register_check(VectorCheck())

    valid_embedding = [0.5] * 768
    report = validator.validate(valid_embedding)

    # Should be ISO format
    assert "T" in report.timestamp
    assert report.timestamp.endswith("Z") or "+" in report.timestamp or "-" in report.timestamp[-6:]


def test_vector_check_all_elements_float():
    """VectorCheck validates all elements are float type."""
    check = VectorCheck()

    # Mix of float and int (int could be valid in Python but we check strict types)
    mixed_embedding = [0.5] * 767 + [1]  # Last element is int

    result = check.validate(mixed_embedding)

    # Should fail because last element is int, not float
    # Note: In practice, Python might coerce int to float, so this tests strict typing
    # You may need to adjust based on actual behavior
    if not isinstance(mixed_embedding[-1], float):
        assert result.passed is False


def test_validator_report_different_failure_messages():
    """Each failed check has distinct message in report."""
    validator = EmbeddingValidator()

    # Add VectorCheck that will fail
    validator.register_check(VectorCheck())

    wrong_dims = [0.5] * 512
    report = validator.validate(wrong_dims)

    assert "vector_format" in report.messages
    assert "Expected 768" in report.messages["vector_format"]


def test_similarity_check_vector_normalization():
    """SimilarityCheck cosine_similarity handles normalization."""
    generator = EmbeddingGenerator()
    check = SimilarityCheck(generator)

    # Zero vectors
    zero_v1 = [0.0, 0.0, 0.0]
    zero_v2 = [0.0, 0.0, 0.0]

    similarity = check._cosine_similarity(zero_v1, zero_v2)

    # Should handle gracefully (return 0 or handle zero division)
    assert isinstance(similarity, float)
    assert 0.0 <= similarity <= 1.0


def test_consistency_check_different_test_codes():
    """ConsistencyCheck uses deterministic test code."""
    generator = EmbeddingGenerator()
    check1 = ConsistencyCheck(generator)
    check2 = ConsistencyCheck(generator)

    dummy_embedding = [0.5] * 768
    result1 = check1.validate(dummy_embedding)
    result2 = check2.validate(dummy_embedding)

    # Both should pass independently
    assert result1.passed is True
    assert result2.passed is True


def test_validator_full_integration_pipeline():
    """Full validation pipeline with all checks works end-to-end."""
    generator = EmbeddingGenerator()
    validator = EmbeddingValidator()

    # Register all checks
    validator.register_check(VectorCheck())
    validator.register_check(SimilarityCheck(generator))
    validator.register_check(ConsistencyCheck(generator))

    # Validate a real embedding from EmbeddingGenerator
    code = "def hello(): return 'world'"
    embedding = generator.embed_code(code)

    report = validator.validate(embedding)

    # Should have 3 checks attempted
    assert len(report.checks_passed) + len(report.checks_failed) == 3
    # Should pass all checks
    assert report.embedding_valid is True
    assert len(report.checks_failed) == 0


def test_batch_validation_with_mixed_validity():
    """Batch validation handles valid and invalid embeddings."""
    validator = EmbeddingValidator()
    validator.register_check(VectorCheck())

    valid_embedding = [0.5] * 768
    invalid_embedding = [0.5] * 512  # Wrong dimensions

    embeddings = [valid_embedding, invalid_embedding, valid_embedding]
    reports = validator.validate_batch(embeddings)

    assert len(reports) == 3
    assert reports[0].embedding_valid is True
    assert reports[1].embedding_valid is False
    assert reports[2].embedding_valid is True
```

**Step 2: Run all tests to verify they pass**

Run: `python3 -m pytest tests/test_embedding_validator.py -v`

Expected: All tests PASSED (30+ tests total)

**Step 3: Check coverage**

Run: `python3 -m pytest tests/test_embedding_validator.py --cov=codesearch.embeddings.validator --cov-report=term-missing`

Expected: ≥95% coverage on validator.py

**Step 4: Commit**

```bash
git add tests/test_embedding_validator.py
git commit -m "feat: Add comprehensive unit tests for embedding validator (Task 5)"
```

---

## Verification Checklist

After implementing all 5 tasks:

```bash
# Run all tests
python3 -m pytest tests/test_embedding_validator.py -v

# Check coverage
python3 -m pytest tests/test_embedding_validator.py --cov=codesearch.embeddings.validator --cov-report=term-missing

# Verify validator can be imported
python3 -c "from codesearch.embeddings import ValidationCheck, VectorCheck, SimilarityCheck, ConsistencyCheck, EmbeddingValidator, ValidationReport; print('All imports successful')"

# Run full project test suite to ensure no regressions
python3 -m pytest tests/ -v
```

Expected results:
- All embedding validator tests passing (30+ tests)
- ≥95% coverage on validator.py
- All project tests passing (70+ tests total)
- No import errors
- No type errors

---

## Summary

This implementation plan creates a composable, extensible embedding validation system with:

- **ValidationCheck interface**: Base class for all validations
- **VectorCheck**: Tensor property validation (dimensions, NaN/Inf, ranges)
- **SimilarityCheck**: Semantic correctness via code pattern pairs
- **ConsistencyCheck**: Determinism validation across runs
- **EmbeddingValidator**: Orchestrator combining all checks
- **ValidationReport**: Detailed results with human-readable summary
- **Comprehensive tests**: 30+ unit tests with ≥95% coverage

The architecture follows SOLID principles (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion) and enables easy addition of new validation checks without modifying existing code.
