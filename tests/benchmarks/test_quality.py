"""Tests for quality benchmarking."""

import pytest

from codesearch.benchmarks.quality import (
    QualityBenchmark,
    QualityMetrics,
    QualityTestCase,
    TestCategoryResult,
    cosine_similarity,
    QUALITY_TEST_SUITE,
)


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        """Similar vectors should have high similarity."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.1, 2.1, 3.1]
        sim = cosine_similarity(vec1, vec2)
        assert sim > 0.99

    def test_zero_vector(self):
        """Zero vector should return 0 similarity."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec1, vec2) == 0.0


class TestQualityTestSuite:
    """Tests for the quality test suite."""

    def test_suite_has_categories(self):
        """Test suite should have multiple categories."""
        assert len(QUALITY_TEST_SUITE) >= 3

    def test_same_function_variants_exists(self):
        """Same function variants category should exist."""
        assert "same_function_variants" in QUALITY_TEST_SUITE
        test_case = QUALITY_TEST_SUITE["same_function_variants"]
        assert test_case.expected_similarity == "high"
        assert len(test_case.pairs) >= 2

    def test_different_functions_exists(self):
        """Different functions category should exist."""
        assert "different_functions" in QUALITY_TEST_SUITE
        test_case = QUALITY_TEST_SUITE["different_functions"]
        assert test_case.expected_similarity == "low"

    def test_cross_language_exists(self):
        """Cross-language category should exist."""
        assert "cross_language_python_js" in QUALITY_TEST_SUITE
        test_case = QUALITY_TEST_SUITE["cross_language_python_js"]
        assert test_case.expected_similarity == "medium"


class TestQualityBenchmark:
    """Tests for QualityBenchmark class."""

    @pytest.fixture
    def mock_embed_func(self):
        """Create a mock embedding function."""
        # Simple mock that returns different embeddings for different code
        def embed(code: str):
            # Create a simple hash-based embedding
            import hashlib
            h = hashlib.md5(code.encode()).hexdigest()
            # Convert hex to floats
            return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
        return embed

    @pytest.fixture
    def constant_embed_func(self):
        """Create an embedding function that returns constant values."""
        def embed(code: str):
            return [0.5] * 16
        return embed

    def test_run_returns_metrics(self, mock_embed_func):
        """Run should return QualityMetrics."""
        benchmark = QualityBenchmark()
        metrics = benchmark.run("test-model", mock_embed_func)

        assert isinstance(metrics, QualityMetrics)
        assert metrics.model_name == "test-model"
        assert 0.0 <= metrics.overall_score <= 1.0

    def test_run_includes_all_categories(self, mock_embed_func):
        """Run should include results for all test categories."""
        benchmark = QualityBenchmark()
        metrics = benchmark.run("test-model", mock_embed_func)

        for key in QUALITY_TEST_SUITE:
            assert key in metrics.test_results

    def test_custom_test_suite(self, mock_embed_func):
        """Should support custom test suite."""
        custom_suite = {
            "custom_test": QualityTestCase(
                name="Custom Test",
                description="Test description",
                pairs=[("def a(): pass", "def b(): pass")],
                expected_similarity="high",
            )
        }
        benchmark = QualityBenchmark(test_suite=custom_suite)
        metrics = benchmark.run("test-model", mock_embed_func)

        assert "custom_test" in metrics.test_results
        assert len(metrics.test_results) == 1

    def test_constant_embeddings_same_similarity(self, constant_embed_func):
        """Constant embeddings should give perfect similarity."""
        benchmark = QualityBenchmark()
        metrics = benchmark.run("test-model", constant_embed_func)

        # All pairs should have similarity 1.0
        for result in metrics.test_results.values():
            assert result.avg_similarity == pytest.approx(1.0)


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""

    def test_to_dict(self):
        """to_dict should convert metrics to dictionary."""
        metrics = QualityMetrics(
            model_name="test-model",
            overall_score=0.85,
            test_results={
                "test1": TestCategoryResult(
                    category_name="Test 1",
                    expected_similarity="high",
                    avg_similarity=0.9,
                    min_similarity=0.85,
                    max_similarity=0.95,
                    pair_similarities=[0.85, 0.9, 0.95],
                    score=0.9,
                )
            },
        )

        result = metrics.to_dict()
        assert result["model_name"] == "test-model"
        assert result["overall_score"] == 0.85
        assert "test1" in result["test_results"]


class TestTestCategoryResult:
    """Tests for TestCategoryResult dataclass."""

    def test_to_dict(self):
        """to_dict should convert result to dictionary."""
        result = TestCategoryResult(
            category_name="Test Category",
            expected_similarity="high",
            avg_similarity=0.9,
            min_similarity=0.85,
            max_similarity=0.95,
            pair_similarities=[0.85, 0.9, 0.95],
            score=0.9,
        )

        d = result.to_dict()
        assert d["category_name"] == "Test Category"
        assert d["expected_similarity"] == "high"
        assert d["avg_similarity"] == 0.9
        assert d["score"] == 0.9


class TestScoreCalculation:
    """Tests for score calculation logic."""

    def test_high_similarity_in_range(self):
        """High similarity in expected range should score 1.0."""
        benchmark = QualityBenchmark()
        score = benchmark._calculate_score(0.9, "high")
        assert score == 1.0

    def test_high_similarity_below_range(self):
        """High similarity below range should penalize."""
        benchmark = QualityBenchmark()
        score = benchmark._calculate_score(0.5, "high")
        assert score < 1.0

    def test_low_similarity_in_range(self):
        """Low similarity in expected range should score 1.0."""
        benchmark = QualityBenchmark()
        score = benchmark._calculate_score(0.3, "low")
        assert score == 1.0

    def test_low_similarity_above_range(self):
        """Low similarity above range should penalize."""
        benchmark = QualityBenchmark()
        score = benchmark._calculate_score(0.8, "low")
        assert score < 1.0

    def test_medium_similarity_in_range(self):
        """Medium similarity in expected range should score 1.0."""
        benchmark = QualityBenchmark()
        score = benchmark._calculate_score(0.65, "medium")
        assert score == 1.0
