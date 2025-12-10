"""Performance benchmarking for embedding models.

Measures:
- Embedding speed (samples/second)
- Memory usage (peak RAM)
- Storage requirements (bytes per embedding)
"""

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Default code samples for benchmarking
DEFAULT_BENCHMARK_SAMPLES = [
    # Short functions
    "def add(a, b): return a + b",
    "def multiply(x, y): return x * y",
    "def is_even(n): return n % 2 == 0",
    # Medium functions
    """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",
    """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
    # Longer function
    """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)""",
    # Class
    """class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, value):
        self.result += value
        return self

    def subtract(self, value):
        self.result -= value
        return self

    def reset(self):
        self.result = 0
        return self""",
    # JavaScript-like
    "function greet(name) { return `Hello, ${name}!`; }",
    # Go-like
    "func main() { fmt.Println('Hello, World!') }",
    # SQL-like
    "SELECT id, name FROM users WHERE active = 1 ORDER BY created_at DESC",
]


@dataclass
class PerformanceMetrics:
    """Performance metrics for an embedding model."""

    model_name: str
    samples_count: int
    total_time_seconds: float
    samples_per_second: float
    avg_time_per_sample_ms: float
    peak_memory_mb: float
    embedding_dimensions: int
    storage_per_embedding_bytes: int
    storage_per_1k_entities_mb: float
    is_api_model: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "samples_count": self.samples_count,
            "total_time_seconds": round(self.total_time_seconds, 3),
            "samples_per_second": round(self.samples_per_second, 2),
            "avg_time_per_sample_ms": round(self.avg_time_per_sample_ms, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "embedding_dimensions": self.embedding_dimensions,
            "storage_per_embedding_bytes": self.storage_per_embedding_bytes,
            "storage_per_1k_entities_mb": round(self.storage_per_1k_entities_mb, 2),
            "is_api_model": self.is_api_model,
        }


class PerformanceBenchmark:
    """Benchmarks embedding model performance."""

    # Bytes per float32
    BYTES_PER_FLOAT = 4

    def __init__(
        self,
        samples: Optional[List[str]] = None,
        warmup_runs: int = 1,
    ):
        """Initialize performance benchmark.

        Args:
            samples: Code samples to benchmark with
            warmup_runs: Number of warmup runs before measurement
        """
        self.samples = samples or DEFAULT_BENCHMARK_SAMPLES
        self.warmup_runs = warmup_runs

    def run(
        self,
        model_name: str,
        embed_func: Callable[[str], List[float]],
        embed_batch_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
        is_api_model: bool = False,
    ) -> PerformanceMetrics:
        """Run performance benchmark for a model.

        Args:
            model_name: Name of the model being tested
            embed_func: Function that takes code string and returns embedding
            embed_batch_func: Optional batch embedding function
            is_api_model: Whether this is an API-based model (skip memory profiling)

        Returns:
            PerformanceMetrics with results
        """
        # Use batch function if available, otherwise embed one by one
        if embed_batch_func:
            return self._run_batch(model_name, embed_batch_func, is_api_model)
        else:
            return self._run_single(model_name, embed_func, is_api_model)

    def _run_batch(
        self,
        model_name: str,
        embed_batch_func: Callable[[List[str]], List[List[float]]],
        is_api_model: bool,
    ) -> PerformanceMetrics:
        """Run benchmark using batch embedding."""
        # Warmup
        for _ in range(self.warmup_runs):
            _ = embed_batch_func(self.samples[:2])

        # Start memory tracking (skip for API models)
        if not is_api_model:
            tracemalloc.start()

        # Measure time
        start_time = time.perf_counter()
        embeddings = embed_batch_func(self.samples)
        end_time = time.perf_counter()

        # Get memory usage
        if not is_api_model:
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_mb = peak_memory / (1024 * 1024)
        else:
            peak_memory_mb = 0.0

        return self._create_metrics(
            model_name=model_name,
            embeddings=embeddings,
            total_time=end_time - start_time,
            peak_memory_mb=peak_memory_mb,
            is_api_model=is_api_model,
        )

    def _run_single(
        self,
        model_name: str,
        embed_func: Callable[[str], List[float]],
        is_api_model: bool,
    ) -> PerformanceMetrics:
        """Run benchmark using single embedding calls."""
        # Warmup
        for _ in range(self.warmup_runs):
            _ = embed_func(self.samples[0])

        # Start memory tracking (skip for API models)
        if not is_api_model:
            tracemalloc.start()

        # Measure time
        start_time = time.perf_counter()
        embeddings = [embed_func(sample) for sample in self.samples]
        end_time = time.perf_counter()

        # Get memory usage
        if not is_api_model:
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_mb = peak_memory / (1024 * 1024)
        else:
            peak_memory_mb = 0.0

        return self._create_metrics(
            model_name=model_name,
            embeddings=embeddings,
            total_time=end_time - start_time,
            peak_memory_mb=peak_memory_mb,
            is_api_model=is_api_model,
        )

    def _create_metrics(
        self,
        model_name: str,
        embeddings: List[List[float]],
        total_time: float,
        peak_memory_mb: float,
        is_api_model: bool,
    ) -> PerformanceMetrics:
        """Create performance metrics from benchmark results."""
        samples_count = len(self.samples)
        dimensions = len(embeddings[0]) if embeddings else 0

        # Calculate storage
        storage_per_embedding = dimensions * self.BYTES_PER_FLOAT
        storage_per_1k = (storage_per_embedding * 1000) / (1024 * 1024)

        return PerformanceMetrics(
            model_name=model_name,
            samples_count=samples_count,
            total_time_seconds=total_time,
            samples_per_second=samples_count / total_time if total_time > 0 else 0,
            avg_time_per_sample_ms=(total_time / samples_count * 1000) if samples_count > 0 else 0,
            peak_memory_mb=peak_memory_mb,
            embedding_dimensions=dimensions,
            storage_per_embedding_bytes=storage_per_embedding,
            storage_per_1k_entities_mb=storage_per_1k,
            is_api_model=is_api_model,
        )
