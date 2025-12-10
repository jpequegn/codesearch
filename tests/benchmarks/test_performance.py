"""Tests for performance benchmarking."""

import pytest

from codesearch.benchmarks.performance import (
    PerformanceBenchmark,
    PerformanceMetrics,
    DEFAULT_BENCHMARK_SAMPLES,
)


class TestDefaultSamples:
    """Tests for default benchmark samples."""

    def test_samples_exist(self):
        """Default samples should exist."""
        assert len(DEFAULT_BENCHMARK_SAMPLES) > 0

    def test_samples_are_strings(self):
        """All samples should be strings."""
        for sample in DEFAULT_BENCHMARK_SAMPLES:
            assert isinstance(sample, str)

    def test_samples_not_empty(self):
        """Samples should not be empty."""
        for sample in DEFAULT_BENCHMARK_SAMPLES:
            assert len(sample.strip()) > 0


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_to_dict(self):
        """to_dict should convert metrics to dictionary."""
        metrics = PerformanceMetrics(
            model_name="test-model",
            samples_count=10,
            total_time_seconds=1.5,
            samples_per_second=6.67,
            avg_time_per_sample_ms=150.0,
            peak_memory_mb=512.0,
            embedding_dimensions=768,
            storage_per_embedding_bytes=3072,
            storage_per_1k_entities_mb=2.93,
            is_api_model=False,
        )

        result = metrics.to_dict()
        assert result["model_name"] == "test-model"
        assert result["samples_count"] == 10
        assert result["embedding_dimensions"] == 768
        assert result["is_api_model"] is False

    def test_to_dict_api_model(self):
        """to_dict should handle API model flag."""
        metrics = PerformanceMetrics(
            model_name="voyage-code-3",
            samples_count=10,
            total_time_seconds=2.0,
            samples_per_second=5.0,
            avg_time_per_sample_ms=200.0,
            peak_memory_mb=0.0,
            embedding_dimensions=1024,
            storage_per_embedding_bytes=4096,
            storage_per_1k_entities_mb=3.91,
            is_api_model=True,
        )

        result = metrics.to_dict()
        assert result["is_api_model"] is True


class TestPerformanceBenchmark:
    """Tests for PerformanceBenchmark class."""

    @pytest.fixture
    def mock_embed_func(self):
        """Create a mock embedding function."""
        def embed(code: str):
            return [0.1] * 768
        return embed

    @pytest.fixture
    def mock_batch_embed_func(self):
        """Create a mock batch embedding function."""
        def embed_batch(codes):
            return [[0.1] * 768 for _ in codes]
        return embed_batch

    def test_run_with_single_embed(self, mock_embed_func):
        """Run should work with single embedding function."""
        benchmark = PerformanceBenchmark(
            samples=["def test(): pass", "def other(): pass"],
            warmup_runs=0,
        )
        metrics = benchmark.run("test-model", mock_embed_func)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.model_name == "test-model"
        assert metrics.samples_count == 2
        assert metrics.embedding_dimensions == 768

    def test_run_with_batch_embed(self, mock_embed_func, mock_batch_embed_func):
        """Run should prefer batch embedding when available."""
        benchmark = PerformanceBenchmark(
            samples=["def test(): pass"],
            warmup_runs=0,
        )
        metrics = benchmark.run(
            "test-model",
            mock_embed_func,
            embed_batch_func=mock_batch_embed_func,
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.embedding_dimensions == 768

    def test_run_measures_time(self, mock_embed_func):
        """Run should measure time."""
        benchmark = PerformanceBenchmark(
            samples=["def test(): pass"],
            warmup_runs=0,
        )
        metrics = benchmark.run("test-model", mock_embed_func)

        assert metrics.total_time_seconds > 0
        assert metrics.samples_per_second > 0
        assert metrics.avg_time_per_sample_ms > 0

    def test_run_api_model_skips_memory(self, mock_embed_func):
        """API models should skip memory profiling."""
        benchmark = PerformanceBenchmark(
            samples=["def test(): pass"],
            warmup_runs=0,
        )
        metrics = benchmark.run(
            "test-model",
            mock_embed_func,
            is_api_model=True,
        )

        assert metrics.is_api_model is True
        assert metrics.peak_memory_mb == 0.0

    def test_custom_samples(self, mock_embed_func):
        """Should support custom samples."""
        custom_samples = ["print('hello')", "x = 1 + 2"]
        benchmark = PerformanceBenchmark(samples=custom_samples, warmup_runs=0)
        metrics = benchmark.run("test-model", mock_embed_func)

        assert metrics.samples_count == 2

    def test_storage_calculation(self, mock_embed_func):
        """Storage should be calculated correctly."""
        benchmark = PerformanceBenchmark(
            samples=["def test(): pass"],
            warmup_runs=0,
        )
        metrics = benchmark.run("test-model", mock_embed_func)

        # 768 dimensions * 4 bytes = 3072 bytes
        assert metrics.storage_per_embedding_bytes == 768 * 4
        # 3072 * 1000 / 1024 / 1024 ≈ 2.93 MB
        assert metrics.storage_per_1k_entities_mb == pytest.approx(2.93, rel=0.01)


class TestStorageCalculations:
    """Tests for storage calculation."""

    def test_bytes_per_float(self):
        """Should use 4 bytes per float32."""
        assert PerformanceBenchmark.BYTES_PER_FLOAT == 4

    def test_256_dim_storage(self):
        """256-dim embeddings should use ~1MB per 1K entities."""
        # 256 * 4 * 1000 / 1024 / 1024 = 0.9765 MB
        expected = (256 * 4 * 1000) / (1024 * 1024)
        assert expected == pytest.approx(0.976, rel=0.01)

    def test_768_dim_storage(self):
        """768-dim embeddings should use ~3MB per 1K entities."""
        # 768 * 4 * 1000 / 1024 / 1024 = 2.93 MB
        expected = (768 * 4 * 1000) / (1024 * 1024)
        assert expected == pytest.approx(2.93, rel=0.01)

    def test_1024_dim_storage(self):
        """1024-dim embeddings should use ~4MB per 1K entities."""
        # 1024 * 4 * 1000 / 1024 / 1024 = 3.90625 MB
        expected = (1024 * 4 * 1000) / (1024 * 1024)
        assert expected == pytest.approx(3.91, rel=0.01)
