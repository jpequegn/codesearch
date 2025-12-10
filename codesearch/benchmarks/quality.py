"""Quality benchmarking for embedding models.

Measures embedding quality through:
- Same-function similarity (variable renaming should maintain similarity)
- Different-function distinction (unrelated functions should have low similarity)
- Cross-language similarity (equivalent code in different languages)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class QualityTestCase:
    """A single quality test case with code pairs."""

    name: str
    description: str
    pairs: List[Tuple[str, str]]
    expected_similarity: str  # "high", "medium", "low"


# Quality test suite with code pairs
QUALITY_TEST_SUITE: Dict[str, QualityTestCase] = {
    "same_function_variants": QualityTestCase(
        name="Same Function Variants",
        description="Similar functions with variable renaming should have HIGH similarity",
        pairs=[
            (
                "def add(a, b): return a + b",
                "def add(x, y): return x + y",
            ),
            (
                "def multiply(num1, num2):\n    result = num1 * num2\n    return result",
                "def multiply(a, b):\n    product = a * b\n    return product",
            ),
            (
                "def is_even(n): return n % 2 == 0",
                "def is_even(number): return number % 2 == 0",
            ),
            (
                "def greet(name): return f'Hello, {name}!'",
                "def greet(person): return f'Hello, {person}!'",
            ),
        ],
        expected_similarity="high",
    ),
    "different_functions": QualityTestCase(
        name="Different Functions",
        description="Unrelated functions should have LOW similarity",
        pairs=[
            (
                "def add(a, b): return a + b",
                "def fetch_user(id): return db.get(id)",
            ),
            (
                "def calculate_tax(amount, rate): return amount * rate",
                "def send_email(to, subject, body): smtp.send(to, subject, body)",
            ),
            (
                "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "def parse_json(text): return json.loads(text)",
            ),
            (
                "def sort_list(items): return sorted(items)",
                "class DatabaseConnection:\n    def __init__(self, host): self.host = host",
            ),
        ],
        expected_similarity="low",
    ),
    "cross_language_python_js": QualityTestCase(
        name="Cross-Language (Python/JS)",
        description="Equivalent functions in Python and JavaScript should have MEDIUM-HIGH similarity",
        pairs=[
            (
                "def greet(name): return f'Hello {name}'",
                "function greet(name) { return `Hello ${name}` }",
            ),
            (
                "def add(a, b): return a + b",
                "function add(a, b) { return a + b; }",
            ),
            (
                "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)",
                "function factorial(n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}",
            ),
            (
                "def filter_even(nums): return [x for x in nums if x % 2 == 0]",
                "function filterEven(nums) { return nums.filter(x => x % 2 === 0); }",
            ),
        ],
        expected_similarity="medium",
    ),
    "cross_language_python_go": QualityTestCase(
        name="Cross-Language (Python/Go)",
        description="Equivalent functions in Python and Go should have MEDIUM similarity",
        pairs=[
            (
                "def add(a, b): return a + b",
                "func add(a, b int) int { return a + b }",
            ),
            (
                "def is_positive(n): return n > 0",
                "func isPositive(n int) bool { return n > 0 }",
            ),
            (
                "def max_value(a, b):\n    if a > b: return a\n    return b",
                "func maxValue(a, b int) int {\n    if a > b { return a }\n    return b\n}",
            ),
        ],
        expected_similarity="medium",
    ),
}


@dataclass
class QualityMetrics:
    """Quality metrics for an embedding model."""

    model_name: str
    test_results: Dict[str, "TestCategoryResult"] = field(default_factory=dict)
    overall_score: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "overall_score": round(self.overall_score, 3),
            "test_results": {
                name: result.to_dict()
                for name, result in self.test_results.items()
            },
        }


@dataclass
class TestCategoryResult:
    """Results for a single test category."""

    category_name: str
    expected_similarity: str
    avg_similarity: float
    min_similarity: float
    max_similarity: float
    pair_similarities: List[float]
    score: float  # How well it matches expected behavior (0-1)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "category_name": self.category_name,
            "expected_similarity": self.expected_similarity,
            "avg_similarity": round(self.avg_similarity, 4),
            "min_similarity": round(self.min_similarity, 4),
            "max_similarity": round(self.max_similarity, 4),
            "score": round(self.score, 3),
        }


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


class QualityBenchmark:
    """Benchmarks embedding model quality using test suites."""

    # Thresholds for expected similarity levels
    SIMILARITY_THRESHOLDS = {
        "high": (0.85, 1.0),    # Expected range for "high" similarity
        "medium": (0.5, 0.85),  # Expected range for "medium" similarity
        "low": (0.0, 0.5),      # Expected range for "low" similarity
    }

    def __init__(
        self,
        test_suite: Optional[Dict[str, QualityTestCase]] = None,
    ):
        """Initialize quality benchmark.

        Args:
            test_suite: Custom test suite, defaults to QUALITY_TEST_SUITE
        """
        self.test_suite = test_suite or QUALITY_TEST_SUITE

    def run(
        self,
        model_name: str,
        embed_func,
    ) -> QualityMetrics:
        """Run quality benchmark for a model.

        Args:
            model_name: Name of the model being tested
            embed_func: Function that takes code string and returns embedding

        Returns:
            QualityMetrics with test results
        """
        metrics = QualityMetrics(model_name=model_name)

        total_score = 0.0
        test_count = 0

        for test_key, test_case in self.test_suite.items():
            result = self._run_test_category(test_case, embed_func)
            metrics.test_results[test_key] = result
            total_score += result.score
            test_count += 1

        if test_count > 0:
            metrics.overall_score = total_score / test_count

        return metrics

    def _run_test_category(
        self,
        test_case: QualityTestCase,
        embed_func,
    ) -> TestCategoryResult:
        """Run tests for a single category."""
        similarities = []

        for code1, code2 in test_case.pairs:
            emb1 = embed_func(code1)
            emb2 = embed_func(code2)
            sim = cosine_similarity(emb1, emb2)
            similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        min_sim = min(similarities) if similarities else 0.0
        max_sim = max(similarities) if similarities else 0.0

        # Calculate score based on expected similarity
        score = self._calculate_score(avg_sim, test_case.expected_similarity)

        return TestCategoryResult(
            category_name=test_case.name,
            expected_similarity=test_case.expected_similarity,
            avg_similarity=avg_sim,
            min_similarity=min_sim,
            max_similarity=max_sim,
            pair_similarities=similarities,
            score=score,
        )

    def _calculate_score(self, similarity: float, expected: str) -> float:
        """Calculate how well similarity matches expected level.

        Returns a score between 0 and 1 indicating how well the
        actual similarity matches the expected range.
        """
        low, high = self.SIMILARITY_THRESHOLDS.get(expected, (0.0, 1.0))

        if low <= similarity <= high:
            # Perfect match - within expected range
            return 1.0
        elif similarity < low:
            # Below expected range
            if expected == "high":
                # For high expected, being lower is bad
                return max(0.0, similarity / low)
            else:
                # For medium/low expected, being lower might be acceptable
                return 0.8
        else:
            # Above expected range
            if expected == "low":
                # For low expected, being higher is bad
                return max(0.0, 1.0 - (similarity - high) / (1.0 - high))
            else:
                # For high/medium expected, being higher is good
                return 1.0
