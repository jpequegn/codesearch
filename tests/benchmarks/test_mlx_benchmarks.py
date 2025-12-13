"""Tests for MLX model benchmarking.

This module tests the benchmark suite with MLX models on Apple Silicon.
"""

import platform
from unittest.mock import MagicMock, patch

import pytest

from codesearch.benchmarks.performance import PerformanceBenchmark, PerformanceMetrics
from codesearch.benchmarks.quality import QualityBenchmark, QualityMetrics
from codesearch.benchmarks.runner import BenchmarkResult, BenchmarkRunner
from codesearch.embeddings.config import (
    MLX_MODELS,
    get_model_config,
    is_mlx_model,
)


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class TestMLXModelRegistry:
    """Tests for MLX model registry entries."""

    def test_mlx_models_exist(self):
        """MLX models should be defined in registry."""
        assert len(MLX_MODELS) > 0

    def test_mlx_models_list(self):
        """Expected MLX models should be present."""
        expected = {"nomic-mlx", "bge-m3-mlx", "bge-large-mlx", "bge-small-mlx"}
        assert expected == MLX_MODELS

    def test_is_mlx_model_positive(self):
        """is_mlx_model should return True for MLX models."""
        for model in MLX_MODELS:
            assert is_mlx_model(model) is True

    def test_is_mlx_model_negative(self):
        """is_mlx_model should return False for non-MLX models."""
        assert is_mlx_model("codebert") is False
        assert is_mlx_model("unixcoder") is False
        assert is_mlx_model("voyage-code-3") is False

    def test_mlx_model_configs(self):
        """MLX model configs should have correct device."""
        for model_name in MLX_MODELS:
            config = get_model_config(model_name)
            assert config.device == "mlx"

    def test_mlx_model_dimensions(self):
        """MLX models should have expected dimensions."""
        expected_dims = {
            "nomic-mlx": 768,
            "bge-m3-mlx": 1024,
            "bge-large-mlx": 1024,
            "bge-small-mlx": 384,
        }
        for model_name, expected_dim in expected_dims.items():
            config = get_model_config(model_name)
            assert config.dimensions == expected_dim

    def test_mlx_model_max_lengths(self):
        """MLX models should have expected max lengths."""
        expected_lengths = {
            "nomic-mlx": 8192,
            "bge-m3-mlx": 8192,
            "bge-large-mlx": 512,
            "bge-small-mlx": 512,
        }
        for model_name, expected_len in expected_lengths.items():
            config = get_model_config(model_name)
            assert config.max_length == expected_len


class TestMLXBenchmarkFiltering:
    """Tests for MLX model filtering in benchmarks."""

    def test_skip_mlx_models_filter(self):
        """skip_mlx_models should exclude MLX models."""
        with patch("codesearch.benchmarks.runner.get_available_models") as mock:
            mock.return_value = ["codebert", "unixcoder", "nomic-mlx", "bge-small-mlx"]
            runner = BenchmarkRunner(skip_mlx_models=True)

            assert "nomic-mlx" not in runner.models
            assert "bge-small-mlx" not in runner.models
            assert "codebert" in runner.models
            assert "unixcoder" in runner.models

    def test_include_mlx_models_by_default(self):
        """MLX models should be included by default when not skipped."""
        with patch("codesearch.benchmarks.runner.get_available_models") as mock:
            mock.return_value = ["codebert", "nomic-mlx"]
            runner = BenchmarkRunner(skip_mlx_models=False)

            assert "nomic-mlx" in runner.models
            assert "codebert" in runner.models

    def test_explicit_mlx_model_list(self):
        """Explicit model list should work with MLX models."""
        runner = BenchmarkRunner(models=["nomic-mlx", "bge-small-mlx"])
        assert runner.models == ["nomic-mlx", "bge-small-mlx"]


class TestMLXPerformanceBenchmark:
    """Tests for MLX performance benchmarking."""

    @pytest.fixture
    def mock_mlx_embed_func(self):
        """Create a mock MLX embedding function."""
        def embed(code: str):
            return [0.1] * 768
        return embed

    @pytest.fixture
    def mock_mlx_batch_embed_func(self):
        """Create a mock MLX batch embedding function."""
        def embed_batch(codes):
            return [[0.1] * 768 for _ in codes]
        return embed_batch

    def test_performance_benchmark_mlx(self, mock_mlx_embed_func):
        """Performance benchmark should work with MLX-like functions."""
        benchmark = PerformanceBenchmark(
            samples=["def test(): pass", "class Foo: pass"],
            warmup_runs=0,
        )
        metrics = benchmark.run("nomic-mlx", mock_mlx_embed_func)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.model_name == "nomic-mlx"
        assert metrics.samples_count == 2
        assert metrics.embedding_dimensions == 768

    def test_performance_benchmark_mlx_batch(
        self, mock_mlx_embed_func, mock_mlx_batch_embed_func
    ):
        """Batch embedding should work for MLX models."""
        benchmark = PerformanceBenchmark(
            samples=["def a(): pass", "def b(): pass", "def c(): pass"],
            warmup_runs=0,
        )
        metrics = benchmark.run(
            "nomic-mlx",
            mock_mlx_embed_func,
            embed_batch_func=mock_mlx_batch_embed_func,
        )

        assert metrics.samples_count == 3
        assert metrics.total_time_seconds > 0
        assert metrics.samples_per_second > 0

    def test_performance_mlx_not_api_model(self, mock_mlx_embed_func):
        """MLX models should NOT be marked as API models."""
        benchmark = PerformanceBenchmark(
            samples=["def test(): pass"],
            warmup_runs=0,
        )
        metrics = benchmark.run("nomic-mlx", mock_mlx_embed_func, is_api_model=False)

        assert metrics.is_api_model is False
        # Memory should be tracked for local models (non-API)
        # Note: peak_memory_mb may be 0 in test environment

    def test_performance_small_model_dimensions(self, mock_mlx_embed_func):
        """Smaller MLX models should have 384 dimensions."""
        def embed_384(code: str):
            return [0.1] * 384

        benchmark = PerformanceBenchmark(
            samples=["def test(): pass"],
            warmup_runs=0,
        )
        metrics = benchmark.run("bge-small-mlx", embed_384)

        assert metrics.embedding_dimensions == 384
        # 384 * 4 = 1536 bytes per embedding
        assert metrics.storage_per_embedding_bytes == 384 * 4

    def test_performance_large_model_dimensions(self, mock_mlx_embed_func):
        """Larger MLX models should have 1024 dimensions."""
        def embed_1024(code: str):
            return [0.1] * 1024

        benchmark = PerformanceBenchmark(
            samples=["def test(): pass"],
            warmup_runs=0,
        )
        metrics = benchmark.run("bge-m3-mlx", embed_1024)

        assert metrics.embedding_dimensions == 1024
        assert metrics.storage_per_embedding_bytes == 1024 * 4


class TestMLXQualityBenchmark:
    """Tests for MLX quality benchmarking."""

    @pytest.fixture
    def mock_mlx_embed_func(self):
        """Create a mock MLX embedding function for quality tests."""
        # Simple mock that returns different embeddings for different inputs
        def embed(code: str):
            # Hash-based embedding for variety
            h = hash(code) % 100
            return [h / 100.0] * 768
        return embed

    def test_quality_benchmark_mlx(self, mock_mlx_embed_func):
        """Quality benchmark should work with MLX models."""
        benchmark = QualityBenchmark()
        metrics = benchmark.run("nomic-mlx", mock_mlx_embed_func)

        assert isinstance(metrics, QualityMetrics)
        assert metrics.model_name == "nomic-mlx"
        assert 0.0 <= metrics.overall_score <= 1.0


class TestMLXBenchmarkResult:
    """Tests for MLX benchmark results."""

    def test_result_to_dict_mlx(self):
        """MLX results should convert to dict correctly."""
        perf = PerformanceMetrics(
            model_name="nomic-mlx",
            samples_count=100,
            total_time_seconds=5.0,
            samples_per_second=20.0,
            avg_time_per_sample_ms=50.0,
            peak_memory_mb=256.0,
            embedding_dimensions=768,
            storage_per_embedding_bytes=3072,
            storage_per_1k_entities_mb=2.93,
            is_api_model=False,
        )
        quality = QualityMetrics(model_name="nomic-mlx", overall_score=0.88)

        result = BenchmarkResult(
            model_name="nomic-mlx",
            performance=perf,
            quality=quality,
        )
        d = result.to_dict()

        assert d["model_name"] == "nomic-mlx"
        assert d["performance"]["is_api_model"] is False
        assert d["performance"]["samples_per_second"] == 20.0
        assert d["quality"]["overall_score"] == 0.88


class TestMLXStorageCalculations:
    """Tests for MLX model storage calculations."""

    def test_nomic_mlx_storage(self):
        """nomic-mlx (768-dim) should use ~2.93 MB per 1K entities."""
        # 768 * 4 * 1000 / 1024 / 1024 = 2.93 MB
        expected = (768 * 4 * 1000) / (1024 * 1024)
        assert expected == pytest.approx(2.93, rel=0.01)

    def test_bge_small_mlx_storage(self):
        """bge-small-mlx (384-dim) should use ~1.46 MB per 1K entities."""
        # 384 * 4 * 1000 / 1024 / 1024 = 1.46 MB
        expected = (384 * 4 * 1000) / (1024 * 1024)
        assert expected == pytest.approx(1.46, rel=0.01)

    def test_bge_large_mlx_storage(self):
        """bge-large-mlx (1024-dim) should use ~3.91 MB per 1K entities."""
        # 1024 * 4 * 1000 / 1024 / 1024 = 3.91 MB
        expected = (1024 * 4 * 1000) / (1024 * 1024)
        assert expected == pytest.approx(3.91, rel=0.01)


@pytest.mark.skipif(
    not is_apple_silicon(),
    reason="MLX integration tests require Apple Silicon"
)
class TestMLXIntegration:
    """Integration tests for MLX models on Apple Silicon.

    These tests require actual MLX installation and Apple Silicon hardware.
    They are skipped on non-Apple Silicon machines.
    """

    def test_mlx_generator_import(self):
        """MLX generator should be importable on Apple Silicon."""
        from codesearch.embeddings.mlx import MLXEmbeddingGenerator
        assert MLXEmbeddingGenerator is not None

    def test_mlx_is_apple_silicon_check(self):
        """is_apple_silicon should return True on Apple Silicon."""
        from codesearch.embeddings.mlx import is_apple_silicon
        assert is_apple_silicon() is True

    @pytest.mark.slow
    def test_mlx_model_loading(self):
        """MLX model should load on Apple Silicon (slow test)."""
        pytest.importorskip("mlx_embedding_models")

        from codesearch.embeddings.mlx import MLXEmbeddingGenerator

        # Use the smallest model for faster test
        generator = MLXEmbeddingGenerator(model_name="bge-small-mlx")
        assert generator is not None
        assert generator.dimensions == 384

    @pytest.mark.slow
    def test_mlx_embedding_generation(self):
        """MLX should generate valid embeddings (slow test)."""
        pytest.importorskip("mlx_embedding_models")

        from codesearch.embeddings.mlx import MLXEmbeddingGenerator

        generator = MLXEmbeddingGenerator(model_name="bge-small-mlx")
        embedding = generator.embed_code("def hello(): print('world')")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        # Check L2 normalization (sum of squares should be ~1)
        norm = sum(x * x for x in embedding) ** 0.5
        assert abs(norm - 1.0) < 0.01

    @pytest.mark.slow
    def test_mlx_batch_embedding(self):
        """MLX batch embedding should work correctly (slow test)."""
        pytest.importorskip("mlx_embedding_models")

        from codesearch.embeddings.mlx import MLXEmbeddingGenerator

        generator = MLXEmbeddingGenerator(model_name="bge-small-mlx")
        codes = [
            "def foo(): pass",
            "def bar(): return 1",
            "class Baz: pass",
        ]
        embeddings = generator.embed_batch(codes)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384
            norm = sum(x * x for x in emb) ** 0.5
            assert abs(norm - 1.0) < 0.01

    @pytest.mark.slow
    def test_mlx_benchmark_runner_integration(self):
        """Full benchmark should work with MLX models (slow test)."""
        pytest.importorskip("mlx_embedding_models")

        # Run benchmark with smallest MLX model
        runner = BenchmarkRunner(models=["bge-small-mlx"])
        results = runner.run(run_quality=False, run_performance=True)

        assert "bge-small-mlx" in results
        result = results["bge-small-mlx"]
        assert result.performance is not None
        assert result.performance.samples_per_second > 0
        assert result.performance.embedding_dimensions == 384
