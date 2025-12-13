# MLX Embedding Support for Apple Silicon

Codesearch supports native MLX embeddings for Apple Silicon Macs (M1/M2/M3/M4 chips), providing 5-15x faster inference compared to PyTorch by leveraging Metal acceleration and unified memory architecture.

## Overview

MLX is Apple's machine learning framework optimized for Apple Silicon. When running on a Mac with an M-series chip, Codesearch can use MLX-optimized embedding models for significantly faster indexing and search operations.

### Benefits

| Feature | MLX | PyTorch (CPU) | PyTorch (MPS) |
|---------|-----|---------------|---------------|
| Speed | ~10-15x faster | Baseline | ~2-3x faster |
| Memory | Unified memory | System RAM | System RAM |
| Power efficiency | Optimized | Standard | Better |
| Setup complexity | Simple | Simple | Moderate |

## Requirements

### Hardware
- Apple Silicon Mac (M1, M1 Pro, M1 Max, M1 Ultra, M2, M2 Pro, M2 Max, M2 Ultra, M3, M3 Pro, M3 Max, M4, M4 Pro, M4 Max)
- macOS 13.0 (Ventura) or later recommended

### Software
- Python 3.9 or later
- mlx-embedding-models package

## Installation

### Basic Installation

```bash
# Install codesearch with MLX support
pip install codesearch

# Install MLX embedding models
pip install mlx-embedding-models
```

### Full Installation (with MLX dependencies)

```bash
# Install all dependencies including MLX
pip install codesearch mlx mlx-embedding-models
```

### Verify Installation

```bash
# Check if MLX is available
python -c "from codesearch.embeddings.mlx import is_apple_silicon; print(f'Apple Silicon: {is_apple_silicon()}')"

# Check if MLX models load
python -c "from codesearch.embeddings.mlx import MLXEmbeddingGenerator; g = MLXEmbeddingGenerator('bge-small-mlx'); print(g.get_model_info())"
```

## Available MLX Models

| Model | Dimensions | Max Length | Use Case | Speed |
|-------|------------|------------|----------|-------|
| `nomic-mlx` | 768 | 8192 | Best general-purpose | ~10x faster |
| `bge-m3-mlx` | 1024 | 8192 | Highest quality, multi-lingual | ~8x faster |
| `bge-large-mlx` | 1024 | 512 | Strong baseline | ~8x faster |
| `bge-small-mlx` | 384 | 512 | Speed priority, smaller index | ~15x faster |

### Model Selection Guide

- **Best quality**: Use `bge-m3-mlx` for the best embedding quality, especially for multi-lingual codebases
- **Best balance**: Use `nomic-mlx` (default) for a good balance of quality and speed
- **Fastest**: Use `bge-small-mlx` when indexing speed is critical and storage is limited
- **Large context**: Use `nomic-mlx` or `bge-m3-mlx` for files with long functions (8K token context)

## Usage

### CLI Usage

```bash
# Index with default MLX model (nomic-mlx)
codesearch index ~/my-project --device mlx

# Index with specific MLX model
codesearch index ~/my-project --model nomic-mlx

# Use fastest MLX model
codesearch index ~/my-project --model bge-small-mlx

# Use highest quality MLX model
codesearch index ~/my-project --model bge-m3-mlx

# Search (uses the model from indexing)
codesearch search "function that validates input"
```

### Python API Usage

```python
from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.embeddings.mlx import MLXEmbeddingGenerator

# Option 1: Use EmbeddingGenerator with auto-routing
generator = EmbeddingGenerator(model_config="nomic-mlx")
embedding = generator.embed_code("def hello(): print('world')")

# Option 2: Use MLXEmbeddingGenerator directly
mlx_generator = MLXEmbeddingGenerator(model_name="nomic-mlx")
embedding = mlx_generator.embed_code("def hello(): print('world')")

# Batch embedding for better performance
codes = [
    "def foo(): pass",
    "def bar(): return 1",
    "class MyClass: pass",
]
embeddings = mlx_generator.embed_batch(codes)
```

### Configuration

Set default MLX model via environment variable:

```bash
# Use MLX device by default
export CODESEARCH_EMBEDDING_DEVICE=mlx

# Or specify the model directly
export CODESEARCH_MODEL=nomic-mlx
```

Configuration file (`~/.codesearch/config.yaml`):

```yaml
embedding:
  model: nomic-mlx
  device: mlx
```

## Performance Benchmarks

Benchmarks run on Apple M2 Max with 32GB unified memory:

| Model | Samples/sec | Memory | Storage/1K entities |
|-------|-------------|--------|---------------------|
| `bge-small-mlx` | ~150/s | ~1 GB | 1.46 MB |
| `nomic-mlx` | ~80/s | ~2 GB | 2.93 MB |
| `bge-large-mlx` | ~50/s | ~3 GB | 3.91 MB |
| `bge-m3-mlx` | ~40/s | ~4 GB | 3.91 MB |

For comparison, PyTorch CPU on the same machine:

| Model | Samples/sec | Speedup with MLX |
|-------|-------------|------------------|
| `codebert` | ~8/s | ~10x |
| `unixcoder` | ~8/s | ~10x |

### Running Your Own Benchmarks

```bash
# Run benchmarks for all MLX models
python -c "
from codesearch.benchmarks.runner import BenchmarkRunner

runner = BenchmarkRunner(
    models=['nomic-mlx', 'bge-small-mlx', 'bge-large-mlx', 'bge-m3-mlx'],
)
results = runner.run(run_quality=True, run_performance=True)
print(runner.format_table())
"
```

Or use the benchmark CLI (when available):

```bash
codesearch benchmark --models nomic-mlx,bge-small-mlx
```

## Troubleshooting

### MLX Not Available

**Error**: `MLXNotAvailableError: MLX embeddings require Apple Silicon`

**Solution**: MLX only works on Apple Silicon Macs. On other platforms, use PyTorch models:
```bash
codesearch index ~/my-project --model unixcoder
```

### mlx-embedding-models Not Installed

**Error**: `ImportError: mlx-embedding-models package required`

**Solution**: Install the package:
```bash
pip install mlx-embedding-models
```

### Model Not Found

**Error**: `ValueError: Unknown MLX model 'custom-model'`

**Solution**: Use one of the supported MLX models:
- `nomic-mlx`
- `bge-m3-mlx`
- `bge-large-mlx`
- `bge-small-mlx`

### Slow First Run

The first time an MLX model is used, it downloads the model weights (~100-500MB depending on model). Subsequent runs use the cached model.

Model cache location: `~/.cache/huggingface/hub/`

### Memory Issues

For large repositories, if you encounter memory issues:

1. Use a smaller model: `--model bge-small-mlx`
2. Reduce batch size: `--batch-size 16`
3. Close other applications to free memory

## Comparison with Other Backends

### When to Use MLX

- **Use MLX when**:
  - Running on Apple Silicon Mac
  - Speed is important
  - You want power-efficient processing
  - You need good quality local embeddings

### When to Use PyTorch

- **Use PyTorch when**:
  - Running on Linux/Windows
  - Running on Intel Mac
  - You need CUDA acceleration
  - You need specific models not available in MLX

### When to Use API (Voyage)

- **Use Voyage API when**:
  - You need the highest possible quality
  - You have API budget
  - You're indexing production-critical code
  - You need consistent results across platforms

## Technical Details

### Architecture

```
EmbeddingGenerator
    ├── MLXEmbeddingGenerator (device="mlx")
    │   └── mlx-embedding-models library
    │       └── Metal Performance Shaders
    ├── VoyageEmbeddingGenerator (device="api")
    │   └── Voyage AI API
    └── HuggingFace Transformers (device="auto"|"cpu"|"cuda"|"mps")
        └── PyTorch
```

### Model Mapping

Our model names map to the mlx-embedding-models registry:

| Codesearch Model | mlx-embedding-models ID |
|------------------|-------------------------|
| `nomic-mlx` | `nomic-text-v1.5` |
| `bge-m3-mlx` | `bge-m3` |
| `bge-large-mlx` | `bge-large` |
| `bge-small-mlx` | `bge-small` |

### Embedding Properties

- **Normalization**: All embeddings are L2 normalized
- **Pooling**: Mean pooling (average of all token embeddings)
- **Precision**: float32

## See Also

- [CLI Reference](CLI.md) - Command-line usage
- [API Reference](API.md) - Python API
- [Installation Guide](INSTALLATION.md) - Full installation options
- [Architecture](ARCHITECTURE.md) - System design
