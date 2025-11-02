# Embedding Quality Assurance Design

**Date**: 2025-11-01
**Issue**: #8
**Component**: 2.4 - Embedding Quality Assurance

## Overview

This document specifies the design for embedding quality assurance - validating that generated embeddings are correct, deterministic, and semantically meaningful. The system validates embedding tensor properties, semantic correctness through pattern testing, and consistency guarantees.

## Requirements

- **Primary Goal**: Validate embedding correctness with composable validation checks
- **Validation Scope**: Vector properties + semantic correctness + deterministic output
- **Testing Strategy**: Unit tests with known code pattern pairs
- **Architecture**: Composable check pipeline with orchestrator
- **Correctness Priority**: Validation > performance optimization
- **Extensibility**: Easy to add new validation checks

## Architecture

### Core Components

#### ValidationCheck (Abstract Base)

Interface that all validation checks must implement:

```python
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
```

#### VectorCheck (Concrete Implementation)

Validates embedding tensor properties:

```python
class VectorCheck(ValidationCheck):
    """Validates embedding vector format and properties."""

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

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

    @property
    def name(self) -> str:
        return "vector_format"
```

**Why these validations?**
- **Dimensions**: CodeBERT must produce exactly 768 dims; wrong size breaks vector DB schema
- **NaN/Infinity**: Indicates model errors, numeric instability, or training failures
- **Value Range**: Detects normalization issues or model drift from expected behavior

#### SimilarityCheck (Concrete Implementation)

Validates semantic correctness through known code pattern pairs:

```python
# Pre-defined test patterns with expected similarity relationships
TEST_PATTERNS = {
    "similar_arithmetic": {
        "code1": "def add(a, b):\n    return a + b",
        "code2": "def subtract(a, b):\n    return a - b",
        "expected_similarity": (0.75, 1.0),  # Should be very similar
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
        "expected_similarity": (0.0, 0.50),  # Should be very different
        "reason": "Different domains: math vs parsing"
    },
    "unrelated_math_network": {
        "code1": "def factorial(n):\n    return n * (n-1)",
        "code2": "def http_request(url):\n    return requests.get(url)",
        "expected_similarity": (0.0, 0.40),  # Should be very different
        "reason": "Different domains: math vs networking"
    }
}

class SimilarityCheck(ValidationCheck):
    """Validates semantic correctness via known code pattern pairs."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        patterns: Dict = TEST_PATTERNS,
        tolerance: float = 0.05
    ):
        self.generator = embedding_generator
        self.patterns = patterns
        self.tolerance = tolerance

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

    @property
    def name(self) -> str:
        return "semantic_correctness"
```

**Why this approach?**
- **Pattern pairs**: Tests real semantic relationships, not just mathematical properties
- **Dual testing**: Similar patterns should cluster, unrelated patterns should diverge
- **Threshold bounds**: Tuned bounds prevent false positives/negatives
- **Tolerance parameter**: Allows for minor variation while catching model drift

#### ConsistencyCheck (Concrete Implementation)

Validates deterministic output:

```python
class ConsistencyCheck(ValidationCheck):
    """Validates that embeddings are deterministic across runs."""

    def __init__(self, embedding_generator: EmbeddingGenerator, runs: int = 3):
        self.generator = embedding_generator
        self.runs = runs

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

    @property
    def name(self) -> str:
        return "deterministic_output"
```

**Why this check?**
- **Determinism guarantee**: Critical for caching; same code must always produce same embedding
- **Reproducibility**: Enables testing, debugging, and version tracking
- **Quality indicator**: Non-determinism signals model instability or initialization issues

#### EmbeddingValidator (Orchestrator)

Runs all checks and aggregates results:

```python
@dataclass
class ValidationResult:
    """Result of a single validation check."""
    passed: bool
    message: str

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
            return f"✗ {len(self.checks_failed)} of {len(self.checks_passed) + len(self.checks_failed)} checks failed"

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

**Design principle:**
- **Separation of concerns**: Each check focuses on one aspect
- **Composability**: Add/remove checks without changing orchestrator
- **Clear reporting**: ValidationReport shows what passed/failed with details

## Integration Points

### With EmbeddingGenerator (Issue #5)
- Uses `generator.embed_code(text)` in SimilarityCheck and ConsistencyCheck
- Validates output is 768-dimensional float list
- Tests for determinism (critical for reproducibility)

### With BatchEmbeddingGenerator (Issue #7)
- Can validate batch output: `validator.validate_batch(embeddings_dict.values())`
- Optional post-batch validation: run checks on sample of generated embeddings
- Could add `embedding_valid` field to batch result summary

### With TextPreparator (Issue #6)
- Not directly used, but validates quality of EmbeddingGenerator which consumes TextPreparator output
- Indirectly ensures text preparation doesn't cause issues

## Testing Strategy

### Unit Tests

**VectorCheck Tests:**
- `test_valid_768_embedding` - Accepts correct dimensions
- `test_invalid_dimensions` - Rejects wrong dimensions
- `test_nan_detection` - Catches NaN values
- `test_infinity_detection` - Catches infinity values
- `test_out_of_range_values` - Catches out-of-bounds values
- `test_valid_range` - Accepts normalized values

**SimilarityCheck Tests:**
- `test_similar_patterns_high_similarity` - Similar patterns above threshold
- `test_unrelated_patterns_low_similarity` - Unrelated patterns below threshold
- `test_all_pattern_pairs` - Validates all test patterns at once
- `test_tolerance_handling` - Respects tolerance parameter
- `test_empty_patterns` - Handles no patterns gracefully

**ConsistencyCheck Tests:**
- `test_deterministic_output` - Same code produces same embedding
- `test_multiple_runs` - Verified across 3+ runs
- `test_different_code_different_embedding` - Different inputs still deterministic

**EmbeddingValidator Tests:**
- `test_validator_initialization` - No checks registered initially
- `test_register_check` - Can add checks
- `test_single_check_pass` - All checks pass → embedding_valid=True
- `test_single_check_fail` - One check fails → embedding_valid=False
- `test_multiple_checks_mixed` - Some pass, some fail
- `test_validate_batch` - Validates multiple embeddings
- `test_summary_output` - Human-readable summary generated

### Test Coverage Target

- ≥95% code coverage on validator classes
- All validation paths tested (pass and fail cases)
- Edge cases: empty embeddings, malformed data, boundary values

## Success Criteria

✓ ValidationCheck interface created and documented
✓ VectorCheck implementation validates tensor properties
✓ SimilarityCheck with test patterns validates semantic correctness
✓ ConsistencyCheck validates deterministic output
✓ EmbeddingValidator orchestrator working correctly
✓ All tests passing (20+ unit tests)
✓ ≥95% code coverage on validation code
✓ ValidationReport provides clear pass/fail with details
✓ Ready for integration with BatchEmbeddingGenerator (Issue #7)

## Files to Create

- `codesearch/embeddings/validator.py` - ValidationCheck, concrete implementations, EmbeddingValidator
- `tests/test_embedding_validator.py` - Comprehensive test suite (20+ tests)
- `codesearch/embeddings/test_patterns.py` - Shared test pattern definitions

## Implementation Plan Structure

The implementation will follow TDD with 5 focused tasks:

1. **ValidationCheck interface + VectorCheck** - Basic structure and first validation
2. **SimilarityCheck with pattern pairs** - Semantic correctness testing
3. **ConsistencyCheck** - Determinism validation
4. **EmbeddingValidator orchestrator** - Bringing it all together
5. **Integration testing** - Full pipeline validation with real components

## Next Steps

1. Create ValidationCheck interface and VectorCheck implementation
2. Implement SimilarityCheck with tuned pattern bounds
3. Implement ConsistencyCheck with determinism verification
4. Create EmbeddingValidator orchestrator with batch support
5. Write comprehensive test suite (20+ tests)
6. Validate with real EmbeddingGenerator output
7. Commit PR for review
8. Proceed to Issue #9: Vector Database (LanceDB) Schema Design
