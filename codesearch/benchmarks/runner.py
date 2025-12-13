"""Benchmark runner that orchestrates quality and performance tests."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from codesearch.benchmarks.performance import PerformanceBenchmark, PerformanceMetrics
from codesearch.benchmarks.quality import QualityBenchmark, QualityMetrics
from codesearch.embeddings.config import get_available_models, get_model_config, is_mlx_model


@dataclass
class BenchmarkResult:
    """Combined benchmark results for a model."""

    model_name: str
    quality: Optional[QualityMetrics] = None
    performance: Optional[PerformanceMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "quality": self.quality.to_dict() if self.quality else None,
            "performance": self.performance.to_dict() if self.performance else None,
        }


class BenchmarkRunner:
    """Runs benchmarks for embedding models."""

    def __init__(
        self,
        models: Optional[List[str]] = None,
        samples: Optional[List[str]] = None,
        skip_api_models: bool = False,
        skip_mlx_models: bool = False,
    ):
        """Initialize benchmark runner.

        Args:
            models: List of model names to benchmark (defaults to all)
            samples: Custom code samples for benchmarking
            skip_api_models: Skip API-based models (e.g., voyage-code-3)
            skip_mlx_models: Skip MLX models (e.g., nomic-mlx, bge-*-mlx)
        """
        self.models = models or self._get_filtered_models(skip_api_models, skip_mlx_models)
        self.samples = samples
        self.quality_benchmark = QualityBenchmark()
        self.performance_benchmark = PerformanceBenchmark(samples=samples)
        self.results: Dict[str, BenchmarkResult] = {}

    def _get_filtered_models(self, skip_api: bool, skip_mlx: bool) -> List[str]:
        """Get list of models to benchmark with filters applied.

        Args:
            skip_api: Skip API-based models
            skip_mlx: Skip MLX-based models

        Returns:
            Filtered list of model names
        """
        all_models = get_available_models()
        filtered = all_models

        if skip_api:
            filtered = [m for m in filtered if get_model_config(m).device != "api"]

        if skip_mlx:
            filtered = [m for m in filtered if not is_mlx_model(m)]

        return filtered

    def run(
        self,
        run_quality: bool = True,
        run_performance: bool = True,
        console: Optional[Console] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run benchmarks for all configured models.

        Args:
            run_quality: Whether to run quality benchmarks
            run_performance: Whether to run performance benchmarks
            console: Rich console for progress output

        Returns:
            Dictionary of model name to BenchmarkResult
        """
        console = console or Console()

        for model_name in self.models:
            console.print(f"\n[bold blue]Benchmarking {model_name}...[/bold blue]")

            try:
                result = self._benchmark_model(
                    model_name,
                    run_quality=run_quality,
                    run_performance=run_performance,
                    console=console,
                )
                self.results[model_name] = result
            except Exception as e:
                console.print(f"[red]Error benchmarking {model_name}: {e}[/red]")
                self.results[model_name] = BenchmarkResult(model_name=model_name)

        return self.results

    def _benchmark_model(
        self,
        model_name: str,
        run_quality: bool,
        run_performance: bool,
        console: Console,
    ) -> BenchmarkResult:
        """Benchmark a single model."""
        from codesearch.embeddings.generator import EmbeddingGenerator

        # Get model config to check if it's API-based
        config = get_model_config(model_name)
        is_api_model = config.device == "api"

        # Create generator
        console.print(f"  Loading model...")
        generator = EmbeddingGenerator(model_config=model_name)

        result = BenchmarkResult(model_name=model_name)

        # Run quality benchmark
        if run_quality:
            console.print(f"  Running quality tests...")
            result.quality = self.quality_benchmark.run(
                model_name=model_name,
                embed_func=generator.embed_code,
            )
            console.print(
                f"  Quality score: [green]{result.quality.overall_score:.3f}[/green]"
            )

        # Run performance benchmark
        if run_performance:
            console.print(f"  Running performance tests...")
            result.performance = self.performance_benchmark.run(
                model_name=model_name,
                embed_func=generator.embed_code,
                embed_batch_func=generator.embed_batch,
                is_api_model=is_api_model,
            )
            console.print(
                f"  Speed: [green]{result.performance.samples_per_second:.1f} samples/s[/green]"
            )

        return result

    def format_table(self, console: Optional[Console] = None) -> Table:
        """Format results as a Rich table.

        Args:
            console: Rich console (unused but kept for consistency)

        Returns:
            Rich Table with benchmark results
        """
        table = Table(title="Embedding Model Benchmark Results")

        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Quality", justify="center")
        table.add_column("Speed", justify="right")
        table.add_column("Memory", justify="right")
        table.add_column("Dims", justify="right")
        table.add_column("Storage/1K", justify="right")

        for model_name, result in self.results.items():
            quality_str = self._format_quality_score(result.quality)
            speed_str = self._format_speed(result.performance)
            memory_str = self._format_memory(result.performance)
            dims_str = str(result.performance.embedding_dimensions) if result.performance else "N/A"
            storage_str = self._format_storage(result.performance)

            table.add_row(
                model_name,
                quality_str,
                speed_str,
                memory_str,
                dims_str,
                storage_str,
            )

        return table

    def _format_quality_score(self, quality: Optional[QualityMetrics]) -> str:
        """Format quality score with stars."""
        if not quality:
            return "N/A"

        score = quality.overall_score
        # Convert 0-1 score to 1-5 stars
        stars = min(5, max(1, int(score * 5) + 1))
        return "⭐" * stars

    def _format_speed(self, perf: Optional[PerformanceMetrics]) -> str:
        """Format speed metric."""
        if not perf:
            return "N/A"
        if perf.is_api_model:
            return "API"
        return f"{perf.samples_per_second:.1f}/s"

    def _format_memory(self, perf: Optional[PerformanceMetrics]) -> str:
        """Format memory metric."""
        if not perf:
            return "N/A"
        if perf.is_api_model:
            return "N/A"
        return f"{perf.peak_memory_mb:.0f} MB"

    def _format_storage(self, perf: Optional[PerformanceMetrics]) -> str:
        """Format storage metric."""
        if not perf:
            return "N/A"
        return f"{perf.storage_per_1k_entities_mb:.1f} MB"

    def format_json(self) -> str:
        """Format results as JSON.

        Returns:
            JSON string with all benchmark results
        """
        output = {
            "benchmark_results": {
                name: result.to_dict()
                for name, result in self.results.items()
            }
        }
        return json.dumps(output, indent=2)

    def save_results(self, path: Path) -> None:
        """Save results to JSON file.

        Args:
            path: Path to save JSON file
        """
        path.write_text(self.format_json())

    @classmethod
    def load_results(cls, path: Path) -> "BenchmarkRunner":
        """Load results from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            BenchmarkRunner with loaded results
        """
        data = json.loads(path.read_text())
        runner = cls(models=[])

        for name, result_data in data.get("benchmark_results", {}).items():
            quality_data = result_data.get("quality")
            perf_data = result_data.get("performance")

            quality = None
            if quality_data:
                quality = QualityMetrics(
                    model_name=quality_data["model_name"],
                    overall_score=quality_data["overall_score"],
                )

            perf = None
            if perf_data:
                perf = PerformanceMetrics(
                    model_name=perf_data["model_name"],
                    samples_count=perf_data["samples_count"],
                    total_time_seconds=perf_data["total_time_seconds"],
                    samples_per_second=perf_data["samples_per_second"],
                    avg_time_per_sample_ms=perf_data["avg_time_per_sample_ms"],
                    peak_memory_mb=perf_data["peak_memory_mb"],
                    embedding_dimensions=perf_data["embedding_dimensions"],
                    storage_per_embedding_bytes=perf_data["storage_per_embedding_bytes"],
                    storage_per_1k_entities_mb=perf_data["storage_per_1k_entities_mb"],
                    is_api_model=perf_data.get("is_api_model", False),
                )

            runner.results[name] = BenchmarkResult(
                model_name=name,
                quality=quality,
                performance=perf,
            )

        return runner
