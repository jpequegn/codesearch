# Text Preparation for Embeddings Design

**Date**: 2025-11-01
**Issue**: #6
**Component**: 2.2 - Embedding Pipeline

## Overview

This document specifies the design for text preparation to optimize code for semantic embedding generation. The system prepares Function and Class objects by intelligently combining source code with docstrings, filtering comments, and truncating to fit embedding model token limits.

## Requirements

- **Primary Goal**: Balanced semantic clarity + token efficiency
- **Comment Strategy**: Smart filtering (keep docstrings + important comments)
- **Whitespace**: Preserve original formatting
- **Architecture**: Single TextPreparator class with cohesive methods

## Architecture

### Core Components

#### TextPreparator Class

Main orchestrator for preparing code text for embeddings:

```python
class TextPreparator:
    """Prepares code text for optimal embedding generation."""

    def __init__(self, tokenizer, max_tokens: int = 512):
        """Initialize with tokenizer and token limit."""
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def prepare_function(self, func: Function) -> str:
        """Prepare a single function for embedding."""

    def prepare_class(self, cls: Class) -> str:
        """Prepare a single class for embedding."""

    def prepare_batch(self, items: List[Union[Function, Class]]) -> List[str]:
        """Prepare multiple items efficiently."""
```

### Preparation Pipeline

The preparation process follows these steps in sequence:

1. **Combine**: Merge docstring + source code
   - Docstring provides semantic context (extracted from Function/Class)
   - Source code provides implementation details
   - Format: `{docstring}\n\n{source_code}` with clear separation

2. **Extract Comments**: Identify and filter important comments
   - Keep docstrings (already extracted, always valuable)
   - Keep block comments (lines starting with `#` spanning multiple lines)
   - Keep inline comments matching patterns: TODO, FIXME, NOTE, BUG, HACK
   - Remove trivial inline comments without important keywords
   - Use regex matching for pattern identification

3. **Normalize**: Preserve original formatting
   - Keep original indentation and spacing
   - Maintain code structure exactly as-is
   - No aggressive whitespace reduction

4. **Truncate**: Fit within token budget
   - Count tokens using embedding model's tokenizer
   - If under max_tokens (512 default): use full prepared text
   - If over: apply truncation strategies in order:
     * Strategy 1: Keep docstring + first N lines of code
     * Strategy 2: Keep docstring + function signature + key parts
     * Strategy 3: Keep docstring only as fallback
   - Always preserve docstring (most semantically important)

5. **Fallback**: Handle unparseable code gracefully
   - If code extraction/parsing fails: use docstring only
   - If no docstring: use function/class name + "unparseable function"
   - Log warning but never fail completely

### Data Flow

```
Function/Class Object (from Parser)
    ↓
Combine docstring + source code
    ↓
Filter comments (smart extraction)
    ↓
Count tokens
    ↓
If exceeds max_tokens: Truncate
    ↓
If fails at any step: Fallback gracefully
    ↓
Prepared text string → to EmbeddingGenerator
```

## Implementation Details

### Docstring + Code Combination

```python
def _combine_text(self, docstring: Optional[str], source_code: str) -> str:
    """Combine docstring and source code with clear separation."""
    if docstring:
        return f"{docstring}\n\n{source_code}"
    return source_code
```

### Smart Comment Filtering

```python
IMPORTANT_COMMENT_PATTERNS = {
    'TODO', 'FIXME', 'NOTE', 'BUG', 'HACK',
    'WARNING', 'DEPRECATED', 'XXX'
}

def _filter_comments(self, code: str) -> str:
    """Keep important comments, remove trivial ones."""
    # Regex to identify important inline comments
    # Keep block comments and important inline comments
    # Remove trivial single-line comments
```

### Token-Aware Truncation

```python
def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
    """Truncate text to stay within token limit while preserving semantics."""
    token_count = len(self.tokenizer.encode(text))
    if token_count <= max_tokens:
        return text

    # Strategy 1: Docstring + first N lines
    # Strategy 2: Docstring + signature + key parts
    # Strategy 3: Docstring only
```

### Graceful Fallback

```python
def prepare_function(self, func: Function) -> str:
    """Prepare function with fallback at each step."""
    try:
        text = self._combine_text(func.docstring, func.source_code)
        text = self._filter_comments(text)
        text = self._truncate_to_tokens(text, self.max_tokens)
        return text
    except Exception as e:
        # Log error
        # Fallback: docstring only, then function name
        return self._get_fallback_text(func)
```

## Integration Points

### With EmbeddingGenerator
- Uses tokenizer from EmbeddingGenerator for consistent token counting
- Output feeds directly to `embed_code()` and `embed_batch()` methods
- Shares same max_tokens configuration

### With Parser
- Works with Function objects (source_code, docstring, name)
- Works with Class objects (source_code, docstring, name)
- Handles both methods and standalone functions

### With Scanner
- Indirectly - prepares code from scanner's discovered files

## Testing Strategy

### Unit Tests

**Basic Preparation**:
- Simple functions → verify docstring + code combined
- Functions without docstrings → verify code-only preparation
- Classes → verify class-level docstring + source code combined

**Comment Filtering**:
- Functions with TODO comments → verify important comments preserved
- Functions with trivial comments → verify trivial comments removed
- Functions with block comments → verify block comments preserved

**Truncation**:
- Long functions → verify truncation keeps docstring + core code
- Functions exceeding token limit → verify final length ≤ max_tokens
- Multiple truncation strategies → verify fallback behavior

**Error Handling**:
- Functions with syntax errors → verify fallback to docstring
- Functions with no docstring and very long signatures → verify name fallback
- Batch processing → verify consistent results with single preparation

**Integration**:
- Prepared text from TextPreparator → verify compatible with EmbeddingGenerator
- Token counts match EmbeddingGenerator's tokenizer

### Edge Cases

- Empty functions (pass only)
- Functions with only docstrings
- Very long function signatures
- Unicode and special characters in comments
- Deeply nested or complex control flow
- Unparseable or malformed code

## Success Criteria

✓ TextPreparator class created with all required methods
✓ Docstring + code combination working correctly
✓ Smart comment filtering extracting important comments
✓ Truncation keeps semantic value while respecting token limits
✓ Graceful fallback for unparseable code
✓ All tests passing with >90% code coverage
✓ Integration verified with EmbeddingGenerator
✓ Real-world validation on P3 codebase
✓ Performance acceptable (preparation overhead <50ms per function)

## Files to Create

- `codesearch/embeddings/text_preparator.py` - TextPreparator implementation
- `tests/test_text_preparator.py` - Comprehensive tests
- `codesearch/embeddings/__init__.py` - Update exports

## Next Steps

1. Implement TextPreparator class with all methods
2. Write comprehensive test suite
3. Validate on real P3 codebase
4. Commit PR for review
5. Proceed to Issue #7 (Component 2.3: Batch Processing Pipeline)
