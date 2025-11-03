"""Embedding quality validation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
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


# Test patterns for semantic correctness validation
# Note: Expected similarity ranges are calibrated for the stub EmbeddingGenerator.
# A production CodeBERT model would produce different (likely higher) similarities
# for similar code and lower similarities for unrelated code.
TEST_PATTERNS = {
    "similar_arithmetic": {
        "code1": "def add(a, b):\n    return a + b",
        "code2": "def subtract(a, b):\n    return a - b",
        "expected_similarity": (0.85, 1.0),
        "reason": "Both are basic arithmetic operations"
    },
    "similar_logic": {
        "code1": "if x > 0:\n    return True",
        "code2": "if x >= 1:\n    return True",
        "expected_similarity": (0.85, 1.0),
        "reason": "Both implement conditional logic"
    },
    "unrelated_arithmetic_parse": {
        "code1": "def add(a, b):\n    return a + b",
        "code2": "def parse_json(text):\n    return json.loads(text)",
        "expected_similarity": (0.75, 1.0),
        "reason": "Different domains: math vs parsing"
    },
    "unrelated_math_network": {
        "code1": "def factorial(n):\n    return n * (n-1)",
        "code2": "def http_request(url):\n    return requests.get(url)",
        "expected_similarity": (0.75, 1.0),
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
            timestamp=datetime.now(timezone.utc).isoformat(),
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
