# Embedding Model Selection & Setup Design

**Date**: 2025-11-01
**Issue**: #5
**Component**: 2.1 - Embedding Pipeline Foundation

## Overview

This document specifies the design for embedding model selection and integration in the Code Intelligence Tool. The system provides a simple, flexible approach to loading and using code embedding models from HuggingFace, with support for different models via configuration.

## Requirements

- **Performance Profile**: Balanced (good speed + good accuracy)
- **Timing**: Batch processing (embeddings generated separately from repository scanning)
- **Architecture Pattern**: Simple + Flexible (pragmatic approach)
- **Default Model**: CodeBERT (microsoft/codebert-base)

## Architecture

### Core Components

#### 1. EmbeddingModel (Data Class)

Represents metadata about an embedding model:

```python
@dataclass
class EmbeddingModel:
    name: str                      # e.g., "codebert-base"
    model_name: str                # HuggingFace model ID
    dimensions: int                # Vector size (e.g., 768)
    max_length: int                # Max input tokens
    device: str = "auto"           # "cpu", "cuda", or "auto"
```

#### 2. EmbeddingGenerator (Main Implementation)

Handles model loading, caching, and embedding generation:

**Key Methods**:
- `__init__(model_config: EmbeddingModel = None)` - Load or download model
- `embed_code(code_text: str) -> List[float]` - Single embedding
- `embed_batch(code_texts: List[str]) -> List[List[float]]` - Batch embeddings
- `get_model_info() -> Dict` - Model metadata

**Features**:
- Automatic HuggingFace model download and local caching
- L2 vector normalization for cosine similarity
- Batch processing for efficiency
- Graceful error handling for encoding issues
- Device auto-detection (GPU/CPU)

### Data Flow

```
Repository Scanner (Issue #1.4)
    ↓
Extracts Functions, Classes, Docstrings
    ↓
Embedding Generator (Issue #2.1) ← You are here
    ↓
Generates 768-dim vectors
    ↓
LanceDB Ingestion (Issue #3.2)
    ↓
Vector Database Storage
```

## Model Loading & Caching

### Initialization Process

1. **Check Local Cache**: Look for model in `~/.codesearch/models/[model-name]/`
2. **Download if Missing**: Use HuggingFace `transformers` library (~300MB for CodeBERT)
3. **Load Components**: Load tokenizer + model into memory
4. **Store Metadata**: Save model dimensions, max_length, version
5. **Configure Device**: Auto-detect GPU, fallback to CPU with warning

### Default Model: CodeBERT

- **HuggingFace ID**: `microsoft/codebert-base`
- **Dimensions**: 768
- **Max Tokens**: 512
- **Languages**: Python, JavaScript, Java, Go
- **Training**: Pre-trained on BigCloneBench (code clone detection)

### Alternative Models (User Configurable)

- `microsoft/codebert-small` - Smaller, faster (384 dims)
- `Salesforce/codet5-base` - Better semantic understanding (768 dims)
- `microsoft/graphcodebert-base` - Graph-aware (768 dims)

### Versioning Strategy

- Model version stored in `~/.codesearch/models/[model-name]/version.txt`
- Support upgrades via: `codesearch model update [model-name]`
- Backward compatibility: old embeddings remain valid, versioned in DB
- No forced upgrades - user-controlled

## Implementation Details

### Tokenization & Normalization

**Tokenization**:
- Split code into tokens using model's tokenizer
- Truncate to max_length tokens (code >512 tokens truncated with `...`)
- Handle special tokens (CLS, SEP) automatically

**Normalization**:
- L2 normalize vectors for cosine similarity in LanceDB
- Ensures consistent similarity scores (0.0 to 1.0 range)

### Error Handling

- **Encoding Errors**: Gracefully handle non-ASCII characters (latin-1 fallback)
- **Syntax Errors**: Code doesn't need to be syntactically valid (pre-trained on raw code)
- **Memory Errors**: Chunk large batches if memory issues
- **Network Errors**: Clear messages if model download fails

### Memory & Performance

- **Inference**: Use `torch.no_grad()` context to skip gradient computation
- **Device Management**: Auto-detect GPU capability, provide CPU fallback
- **Batch Processing**: Significantly faster than single embeddings (~10x)
- **Model Caching**: Keep model in memory during batch operations

## Integration Points

### With Code Parser (Issue #1.x)

- Takes extracted `Function.source_code` or `Function.docstring`
- Takes extracted `Class.source_code` or `Class.docstring`
- Generates 768-dimensional vectors
- Vectors stored alongside code in database

### With Batch Processor (Issue #2.3)

- Batch generator orchestrates embedding generation
- Calls `embed_batch()` for efficiency
- Updates database with embeddings

### With LanceDB (Issue #3.1)

- Embeddings are core to vector database schema
- Enables semantic similarity search
- Supports approximate nearest neighbor queries

## Testing Strategy

### Unit Tests

- Model loading and initialization
- Single embedding generation
- Batch embedding generation
- Error handling (invalid input, encoding errors)
- Device auto-detection

### Integration Tests

- Real functions from P3 codebase → embedding generation
- Embedding consistency (same input → same output)
- Batch vs single comparison
- Model information retrieval

### Edge Cases

- Empty code strings
- Very long code (>512 tokens)
- Non-ASCII characters
- Code with syntax errors
- Special characters and Unicode

### Performance Tests

- Embedding generation speed (should be <50ms per function)
- Batch processing speedup (verify 10x+ improvement)
- Memory usage during batch operations
- Model download and cache behavior

## Files to Create

- `codesearch/embeddings/model.py` - EmbeddingModel dataclass
- `codesearch/embeddings/generator.py` - EmbeddingGenerator implementation
- `codesearch/embeddings/__init__.py` - Module exports
- `tests/test_embedding_generator.py` - Comprehensive tests
- `config/embedding_models.yaml` - Model configuration

## Success Criteria

✓ CodeBERT model loads and generates embeddings
✓ Local caching prevents re-downloads
✓ Batch processing is functional and faster than single
✓ Device auto-detection works (GPU preferred, CPU fallback)
✓ Error handling for edge cases is graceful
✓ All tests passing with >90% code coverage
✓ Integration with parser verified on P3 codebase

## Next Steps

1. Implement EmbeddingGenerator with CodeBERT
2. Create comprehensive tests
3. Validate on P3 codebase
4. Commit PR for review
5. Proceed to Issue #2.2 (Text Preparation) and #2.3 (Batch Processing)
