"""Tests for benchmark runner."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codesearch.benchmarks.runner import (
    BenchmarkResult,
    BenchmarkRunner,
)
from codesearch.benchmarks.performance import PerformanceMetrics
from codesearch.benchmarks.quality import QualityMetrics


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_to_dict_empty(self):
        """to_dict should work with empty results."""
        result = BenchmarkResult(model_name="test-model")
        d = result.to_dict()

        assert d["model_name"] == "test-model"
        assert d["quality"] is None
        assert d["performance"] is None

    def test_to_dict_with_quality(self):
        """to_dict should include quality metrics."""
        quality = QualityMetrics(model_name="test-model", overall_score=0.85)
        result = BenchmarkResult(model_name="test-model", quality=quality)
        d = result.to_dict()

        assert d["quality"] is not None
        assert d["quality"]["overall_score"] == 0.85

    def test_to_dict_with_performance(self):
        """to_dict should include performance metrics."""
        perf = PerformanceMetrics(
            model_name="test-model",
            samples_count=10,
            total_time_seconds=1.0,
            samples_per_second=10.0,
            avg_time_per_sample_ms=100.0,
            peak_memory_mb=512.0,
            embedding_dimensions=768,
            storage_per_embedding_bytes=3072,
            storage_per_1k_entities_mb=2.93,
        )
        result = BenchmarkResult(model_name="test-model", performance=perf)
        d = result.to_dict()

        assert d["performance"] is not None
        assert d["performance"]["samples_per_second"] == 10.0


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    @pytest.fixture
    def mock_generator(self):
        """Create a mock EmbeddingGenerator."""
        mock = MagicMock()
        mock.embed_code.return_value = [0.1] * 768
        mock.embed_batch.return_value = [[0.1] * 768 for _ in range(10)]
        return mock

    def test_init_default_models(self):
        """Should use local models by default."""
        with patch("codesearch.benchmarks.runner.get_available_models") as mock:
            mock.return_value = ["codebert", "unixcoder", "voyage-code-3"]
            with patch("codesearch.benchmarks.runner.get_model_config") as mock_config:
                # Make voyage-code-3 an API model
                def get_config(name):
                    config = MagicMock()
                    config.device = "api" if name == "voyage-code-3" else "auto"
                    return config
                mock_config.side_effect = get_config

                runner = BenchmarkRunner(skip_api_models=True)
                assert "voyage-code-3" not in runner.models
                assert "codebert" in runner.models

    def test_init_with_custom_models(self):
        """Should use custom model list when provided."""
        runner = BenchmarkRunner(models=["codebert"])
        assert runner.models == ["codebert"]

    def test_format_json(self):
        """format_json should return valid JSON."""
        runner = BenchmarkRunner(models=[])
        runner.results["test-model"] = BenchmarkResult(model_name="test-model")

        json_str = runner.format_json()
        data = json.loads(json_str)

        assert "benchmark_results" in data
        assert "test-model" in data["benchmark_results"]

    def test_format_table(self):
        """format_table should return a Rich Table."""
        from rich.table import Table

        runner = BenchmarkRunner(models=[])
        perf = PerformanceMetrics(
            model_name="test-model",
            samples_count=10,
            total_time_seconds=1.0,
            samples_per_second=10.0,
            avg_time_per_sample_ms=100.0,
            peak_memory_mb=512.0,
            embedding_dimensions=768,
            storage_per_embedding_bytes=3072,
            storage_per_1k_entities_mb=2.93,
        )
        runner.results["test-model"] = BenchmarkResult(
            model_name="test-model",
            performance=perf,
        )

        table = runner.format_table()
        assert isinstance(table, Table)

    def test_save_and_load_results(self):
        """Should save and load results from JSON."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "results.json"

            # Create runner with results
            runner = BenchmarkRunner(models=[])
            perf = PerformanceMetrics(
                model_name="test-model",
                samples_count=10,
                total_time_seconds=1.0,
                samples_per_second=10.0,
                avg_time_per_sample_ms=100.0,
                peak_memory_mb=512.0,
                embedding_dimensions=768,
                storage_per_embedding_bytes=3072,
                storage_per_1k_entities_mb=2.93,
            )
            quality = QualityMetrics(model_name="test-model", overall_score=0.85)
            runner.results["test-model"] = BenchmarkResult(
                model_name="test-model",
                performance=perf,
                quality=quality,
            )

            # Save
            runner.save_results(path)
            assert path.exists()

            # Load
            loaded = BenchmarkRunner.load_results(path)
            assert "test-model" in loaded.results
            assert loaded.results["test-model"].performance.samples_per_second == 10.0
            assert loaded.results["test-model"].quality.overall_score == 0.85


class TestQualityScoreFormatting:
    """Tests for quality score formatting."""

    def test_format_quality_score_none(self):
        """Should handle None quality."""
        runner = BenchmarkRunner(models=[])
        result = runner._format_quality_score(None)
        assert result == "N/A"

    def test_format_quality_score_high(self):
        """High score should get many stars."""
        runner = BenchmarkRunner(models=[])
        quality = QualityMetrics(model_name="test", overall_score=0.95)
        result = runner._format_quality_score(quality)
        assert "⭐" in result
        assert result.count("⭐") >= 4

    def test_format_quality_score_low(self):
        """Low score should get few stars."""
        runner = BenchmarkRunner(models=[])
        quality = QualityMetrics(model_name="test", overall_score=0.2)
        result = runner._format_quality_score(quality)
        assert "⭐" in result
        assert result.count("⭐") <= 2


class TestPerformanceFormatting:
    """Tests for performance metric formatting."""

    def test_format_speed_none(self):
        """Should handle None performance."""
        runner = BenchmarkRunner(models=[])
        result = runner._format_speed(None)
        assert result == "N/A"

    def test_format_speed_api_model(self):
        """API models should show 'API'."""
        runner = BenchmarkRunner(models=[])
        perf = PerformanceMetrics(
            model_name="test",
            samples_count=10,
            total_time_seconds=1.0,
            samples_per_second=10.0,
            avg_time_per_sample_ms=100.0,
            peak_memory_mb=0.0,
            embedding_dimensions=1024,
            storage_per_embedding_bytes=4096,
            storage_per_1k_entities_mb=3.91,
            is_api_model=True,
        )
        result = runner._format_speed(perf)
        assert result == "API"

    def test_format_speed_local_model(self):
        """Local models should show samples/sec."""
        runner = BenchmarkRunner(models=[])
        perf = PerformanceMetrics(
            model_name="test",
            samples_count=10,
            total_time_seconds=1.0,
            samples_per_second=15.5,
            avg_time_per_sample_ms=64.5,
            peak_memory_mb=512.0,
            embedding_dimensions=768,
            storage_per_embedding_bytes=3072,
            storage_per_1k_entities_mb=2.93,
            is_api_model=False,
        )
        result = runner._format_speed(perf)
        assert "15.5" in result
        assert "/s" in result

    def test_format_memory_api_model(self):
        """API models should show 'N/A' for memory."""
        runner = BenchmarkRunner(models=[])
        perf = PerformanceMetrics(
            model_name="test",
            samples_count=10,
            total_time_seconds=1.0,
            samples_per_second=10.0,
            avg_time_per_sample_ms=100.0,
            peak_memory_mb=0.0,
            embedding_dimensions=1024,
            storage_per_embedding_bytes=4096,
            storage_per_1k_entities_mb=3.91,
            is_api_model=True,
        )
        result = runner._format_memory(perf)
        assert result == "N/A"
