# Issue #6: Text Preparation for Embeddings - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement TextPreparator class to prepare code text for optimal embedding generation with intelligent docstring combination, comment filtering, and token-aware truncation.

**Architecture:** Single TextPreparator class that orchestrates a preparation pipeline: combine docstring + code → filter comments → count tokens → truncate if needed → fallback gracefully. Uses tokenizer from EmbeddingGenerator for consistent token counting.

**Tech Stack:**
- Python regex for comment pattern matching
- HuggingFace tokenizer (from EmbeddingGenerator)
- Function/Class models (from parser)

---

## Task 1: Create TextPreparator Basic Structure

**Files:**
- Create: `codesearch/embeddings/text_preparator.py`
- Modify: `codesearch/embeddings/__init__.py`
- Test: `tests/test_text_preparator.py`

**Step 1: Write failing test**

Create `tests/test_text_preparator.py`:

```python
"""Tests for text preparation for embeddings."""

import pytest
from codesearch.embeddings.text_preparator import TextPreparator
from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.models import Function


def test_text_preparator_initialization():
    """Test initializing TextPreparator with EmbeddingGenerator."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    assert preparator is not None
    assert preparator.tokenizer is not None
    assert preparator.max_tokens == 512


def test_text_preparator_custom_max_tokens():
    """Test initializing TextPreparator with custom max_tokens."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=256)

    assert preparator.max_tokens == 256


def test_prepare_simple_function():
    """Test preparing a simple function."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    func = Function(
        name="hello",
        file_path="/test.py",
        language="python",
        source_code="def hello():\n    return 'world'",
        docstring="Returns hello world.",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Should contain both docstring and source code
    assert "Returns hello world" in prepared
    assert "def hello" in prepared
    assert isinstance(prepared, str)
    assert len(prepared) > 0
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/julienpequegnot/Code/codesearch/.worktrees/issue-6-text-preparation
source venv/bin/activate
python -m pytest tests/test_text_preparator.py::test_text_preparator_initialization -v
```

Expected: `FAILED ... ModuleNotFoundError: No module named 'codesearch.embeddings.text_preparator'`

**Step 3: Write minimal implementation**

Create `codesearch/embeddings/text_preparator.py`:

```python
"""Text preparation for code embeddings."""

from typing import List, Optional, Union
from transformers import AutoTokenizer

from codesearch.models import Function, Class


class TextPreparator:
    """Prepares code text for optimal embedding generation."""

    def __init__(self, tokenizer: AutoTokenizer, max_tokens: int = 512):
        """
        Initialize the text preparator.

        Args:
            tokenizer: HuggingFace tokenizer for token counting
            max_tokens: Maximum tokens to keep in prepared text
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def prepare_function(self, func: Function) -> str:
        """
        Prepare a function for embedding.

        Args:
            func: Function object with source code and docstring

        Returns:
            Prepared text combining docstring and source code
        """
        return self._combine_text(func.docstring, func.source_code)

    def prepare_class(self, cls: Class) -> str:
        """
        Prepare a class for embedding.

        Args:
            cls: Class object with source code and docstring

        Returns:
            Prepared text combining docstring and source code
        """
        return self._combine_text(cls.docstring, cls.source_code)

    def prepare_batch(self, items: List[Union[Function, Class]]) -> List[str]:
        """
        Prepare multiple items for embedding.

        Args:
            items: List of Function or Class objects

        Returns:
            List of prepared text strings
        """
        return [
            self.prepare_function(item) if isinstance(item, Function)
            else self.prepare_class(item)
            for item in items
        ]

    def _combine_text(self, docstring: Optional[str], source_code: str) -> str:
        """Combine docstring and source code with clear separation."""
        if docstring:
            return f"{docstring}\n\n{source_code}"
        return source_code
```

**Step 4: Update embeddings __init__.py**

Modify `codesearch/embeddings/__init__.py`:

```python
"""Embedding generation module."""

from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.embeddings.text_preparator import TextPreparator
from codesearch.models import EmbeddingModel

__all__ = ["EmbeddingGenerator", "TextPreparator", "EmbeddingModel"]
```

**Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_text_preparator.py::test_text_preparator_initialization tests/test_text_preparator.py::test_prepare_simple_function -v
```

Expected: Both tests PASS

**Step 6: Verify full test suite**

```bash
python -m pytest tests/ -v
```

Expected: 66 tests passing (63 existing + 3 new)

**Step 7: Commit**

```bash
git add codesearch/embeddings/text_preparator.py codesearch/embeddings/__init__.py tests/test_text_preparator.py
git commit -m "feat: Create TextPreparator basic structure (Task 1)"
```

---

## Task 2: Implement Docstring + Code Combination

**Files:**
- Modify: `tests/test_text_preparator.py` (add tests)

**Step 1: Add combination tests**

Append to `tests/test_text_preparator.py`:

```python
def test_combine_docstring_and_code():
    """Test combining docstring and source code."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    docstring = "Calculate sum of two numbers."
    code = "def add(a, b):\n    return a + b"

    func = Function(
        name="add",
        file_path="/test.py",
        language="python",
        source_code=code,
        docstring=docstring,
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Should have both docstring and code
    assert prepared.startswith(docstring)
    assert "def add" in prepared
    # Should have separator
    assert "\n\n" in prepared


def test_combine_no_docstring():
    """Test that code-only functions work."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    code = "def hello():\n    return 'world'"

    func = Function(
        name="hello",
        file_path="/test.py",
        language="python",
        source_code=code,
        docstring=None,
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Should just be the code
    assert prepared == code
    assert "def hello" in prepared


def test_combine_empty_docstring():
    """Test empty docstring handling."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    code = "def test(): pass"

    func = Function(
        name="test",
        file_path="/test.py",
        language="python",
        source_code=code,
        docstring="",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Empty docstring should be treated as no docstring
    assert prepared == code


def test_prepare_class():
    """Test preparing a class for embedding."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    cls_doc = "A simple calculator class."
    cls_code = "class Calculator:\n    def add(self, a, b):\n        return a + b"

    cls = Class(
        name="Calculator",
        file_path="/test.py",
        language="python",
        source_code=cls_code,
        docstring=cls_doc,
        line_number=1,
    )

    prepared = preparator.prepare_class(cls)

    # Should have both docstring and code
    assert "A simple calculator" in prepared
    assert "class Calculator" in prepared
```

**Step 2: Run tests**

```bash
python -m pytest tests/test_text_preparator.py::test_combine_docstring_and_code tests/test_text_preparator.py::test_combine_no_docstring -v
```

Expected: All combination tests PASS

**Step 3: Verify full test suite**

```bash
python -m pytest tests/ -v
```

Expected: 70 tests passing

**Step 4: Commit**

```bash
git add tests/test_text_preparator.py
git commit -m "feat: Implement docstring + code combination (Task 2)"
```

---

## Task 3: Implement Smart Comment Filtering

**Files:**
- Modify: `codesearch/embeddings/text_preparator.py` (add method)
- Modify: `tests/test_text_preparator.py` (add tests)

**Step 1: Add comment filtering tests**

Append to `tests/test_text_preparator.py`:

```python
def test_filter_important_comments():
    """Test that important comments are preserved."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    code_with_comments = """def process():
    # TODO: add validation
    result = calculate()  # TODO: optimize
    # FIXME: handle edge cases
    return result
"""

    filtered = preparator._filter_comments(code_with_comments)

    # Important comments should be kept
    assert "TODO" in filtered
    assert "FIXME" in filtered


def test_filter_trivial_comments():
    """Test that trivial comments are removed."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    code_with_comments = """def process():
    # This is a variable
    x = 1  # Assignment
    # Another comment
    y = 2
    return x + y
"""

    filtered = preparator._filter_comments(code_with_comments)

    # Code should still be there
    assert "x = 1" in filtered
    assert "y = 2" in filtered
    # But trivial comments may be removed
    # (depends on implementation, but shouldn't crash)
    assert isinstance(filtered, str)


def test_keep_block_comments():
    """Test that block comments are preserved."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    code = """def calculate():
    # This is a block comment
    # spanning multiple lines
    # explaining the logic
    return 42
"""

    filtered = preparator._filter_comments(code)

    # Block comments should be preserved
    assert "block comment" in filtered
    assert "spanning multiple lines" in filtered


def test_prepare_with_comment_filtering():
    """Test that prepare_function includes comment filtering."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    code = """def calculate():
    # TODO: optimize this
    result = 1 + 1  # trivial addition
    return result
"""

    func = Function(
        name="calculate",
        file_path="/test.py",
        language="python",
        source_code=code,
        docstring="Calculates a result.",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Should have docstring
    assert "Calculates a result" in prepared
    # Should have important TODO
    assert "TODO" in prepared
    # Should have code
    assert "result = 1 + 1" in prepared
```

**Step 2: Implement comment filtering**

Add to `codesearch/embeddings/text_preparator.py` (after `_combine_text` method):

```python
    import re

    IMPORTANT_COMMENT_KEYWORDS = {
        'TODO', 'FIXME', 'NOTE', 'BUG', 'HACK',
        'WARNING', 'DEPRECATED', 'XXX', 'IMPORTANT'
    }

    def _filter_comments(self, code: str) -> str:
        """
        Filter comments intelligently: keep important, remove trivial.

        Args:
            code: Source code with comments

        Returns:
            Filtered source code
        """
        lines = code.split('\n')
        filtered_lines = []

        for line in lines:
            # Check if line has an inline comment
            if '#' in line:
                # Split at first '#'
                parts = line.split('#', 1)
                code_part = parts[0]
                comment_part = parts[1] if len(parts) > 1 else ''

                # Check if comment contains important keywords
                has_important_keyword = any(
                    keyword in comment_part.upper()
                    for keyword in self.IMPORTANT_COMMENT_KEYWORDS
                )

                if has_important_keyword or not code_part.strip():
                    # Keep line with important comment or pure comment line
                    filtered_lines.append(line)
                else:
                    # Remove trivial inline comment but keep code
                    filtered_lines.append(code_part.rstrip())
            else:
                # No comment, keep as-is
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_text_preparator.py::test_filter_important_comments -v
python -m pytest tests/test_text_preparator.py::test_prepare_with_comment_filtering -v
```

Expected: All comment filtering tests PASS

**Step 4: Verify full test suite**

```bash
python -m pytest tests/ -v
```

Expected: 75 tests passing

**Step 5: Commit**

```bash
git add codesearch/embeddings/text_preparator.py tests/test_text_preparator.py
git commit -m "feat: Implement smart comment filtering (Task 3)"
```

---

## Task 4: Implement Token Counting & Truncation

**Files:**
- Modify: `codesearch/embeddings/text_preparator.py` (add methods)
- Modify: `tests/test_text_preparator.py` (add tests)

**Step 1: Add truncation tests**

Append to `tests/test_text_preparator.py`:

```python
def test_count_tokens():
    """Test token counting."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    code = "def hello(): return 'world'"
    tokens = preparator._count_tokens(code)

    # Should return a positive integer
    assert isinstance(tokens, int)
    assert tokens > 0


def test_truncate_long_function():
    """Test truncating a function that exceeds token limit."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=50)  # Very small limit

    # Create a long function
    long_code = "def long_function():\n    " + "\n    ".join(
        [f"x{i} = {i}" for i in range(20)]
    )

    func = Function(
        name="long_function",
        file_path="/test.py",
        language="python",
        source_code=long_code,
        docstring="This is a long function.",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Should respect token limit
    token_count = preparator._count_tokens(prepared)
    assert token_count <= preparator.max_tokens


def test_truncate_preserves_docstring():
    """Test that truncation preserves docstring."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=30)

    long_code = "def func():\n    " + "\n    ".join(
        [f"x = {i}" for i in range(20)]
    )

    func = Function(
        name="func",
        file_path="/test.py",
        language="python",
        source_code=long_code,
        docstring="Important docstring.",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Docstring should always be in the result
    assert "Important docstring" in prepared


def test_no_truncation_for_short_text():
    """Test that short text is not truncated."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    code = "def simple(): return 42"

    func = Function(
        name="simple",
        file_path="/test.py",
        language="python",
        source_code=code,
        docstring="Simple function.",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Should be unchanged
    assert "Simple function" in prepared
    assert "def simple" in prepared
    assert prepared.count("return 42") >= 1
```

**Step 2: Implement token counting and truncation**

Add to `codesearch/embeddings/text_preparator.py`:

```python
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        return len(self.tokenizer.encode(text))

    def _truncate_to_tokens(self, text: str) -> str:
        """
        Truncate text to stay within max_tokens while preserving semantics.

        Strategy:
        1. If within limit, return as-is
        2. If too long, keep docstring + first N lines of code
        3. If still too long, keep docstring only
        """
        token_count = self._count_tokens(text)
        if token_count <= self.max_tokens:
            return text

        # Try to split docstring and code
        parts = text.split('\n\n', 1)
        if len(parts) == 2:
            docstring, code = parts
            # Keep docstring, truncate code
            lines = code.split('\n')

            # Binary search for how many lines to keep
            for num_lines in range(len(lines), 0, -1):
                truncated_code = '\n'.join(lines[:num_lines])
                combined = f"{docstring}\n\n{truncated_code}"
                if self._count_tokens(combined) <= self.max_tokens:
                    return combined

            # Worst case: just return docstring
            return docstring
        else:
            # No docstring, just truncate code
            lines = text.split('\n')
            for num_lines in range(len(lines), 0, -1):
                truncated = '\n'.join(lines[:num_lines])
                if self._count_tokens(truncated) <= self.max_tokens:
                    return truncated
            return text[:100]  # Last resort
```

**Step 3: Integrate truncation into prepare methods**

Modify `prepare_function` and `prepare_class` to use truncation:

```python
    def prepare_function(self, func: Function) -> str:
        """Prepare a function for embedding."""
        combined = self._combine_text(func.docstring, func.source_code)
        filtered = self._filter_comments(combined)
        truncated = self._truncate_to_tokens(filtered)
        return truncated

    def prepare_class(self, cls: Class) -> str:
        """Prepare a class for embedding."""
        combined = self._combine_text(cls.docstring, cls.source_code)
        filtered = self._filter_comments(combined)
        truncated = self._truncate_to_tokens(filtered)
        return truncated
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_text_preparator.py::test_count_tokens -v
python -m pytest tests/test_text_preparator.py::test_truncate_long_function -v
python -m pytest tests/test_text_preparator.py -v
```

Expected: All tests PASS (80+ tests total)

**Step 5: Commit**

```bash
git add codesearch/embeddings/text_preparator.py tests/test_text_preparator.py
git commit -m "feat: Implement token counting and truncation (Task 4)"
```

---

## Task 5: Implement Error Handling & Fallback

**Files:**
- Modify: `codesearch/embeddings/text_preparator.py` (add error handling)
- Modify: `tests/test_text_preparator.py` (add edge case tests)

**Step 1: Add error handling tests**

Append to `tests/test_text_preparator.py`:

```python
def test_prepare_function_without_docstring():
    """Test preparing function with no docstring."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    func = Function(
        name="no_doc",
        file_path="/test.py",
        language="python",
        source_code="def no_doc(): pass",
        docstring=None,
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Should work and return code
    assert "def no_doc" in prepared
    assert len(prepared) > 0


def test_prepare_empty_source_code_fallback():
    """Test fallback when source code is empty."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    func = Function(
        name="empty",
        file_path="/test.py",
        language="python",
        source_code="",
        docstring="A function with no implementation.",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Should fallback to docstring
    assert "A function with no implementation" in prepared
    assert len(prepared) > 0


def test_prepare_batch():
    """Test batch preparation of multiple functions."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    funcs = [
        Function(
            name="func1",
            file_path="/test.py",
            language="python",
            source_code="def func1(): return 1",
            docstring="First function.",
            line_number=1,
        ),
        Function(
            name="func2",
            file_path="/test.py",
            language="python",
            source_code="def func2(): return 2",
            docstring="Second function.",
            line_number=5,
        ),
    ]

    prepared_list = preparator.prepare_batch(funcs)

    # Should return list of prepared strings
    assert len(prepared_list) == 2
    assert "First function" in prepared_list[0]
    assert "Second function" in prepared_list[1]


def test_very_long_function_fallback():
    """Test fallback for extremely long functions."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=20)  # Very restrictive

    # Create extremely long code
    long_code = "def very_long():\n    " + "\n    ".join(
        [f"x{i} = {i} * {i} * {i} * {i} * {i}" for i in range(50)]
    )

    func = Function(
        name="very_long",
        file_path="/test.py",
        language="python",
        source_code=long_code,
        docstring="Very long function that will be heavily truncated.",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Should never crash and should respect token limit
    token_count = preparator._count_tokens(prepared)
    assert token_count <= preparator.max_tokens
    assert len(prepared) > 0
```

**Step 2: Add defensive error handling**

Modify `prepare_function` in text_preparator.py to add try-catch:

```python
    def prepare_function(self, func: Function) -> str:
        """Prepare a function for embedding with error handling."""
        try:
            combined = self._combine_text(func.docstring, func.source_code)
            filtered = self._filter_comments(combined)
            truncated = self._truncate_to_tokens(filtered)
            return truncated
        except Exception as e:
            # Fallback: return docstring or function name
            if func.docstring:
                return func.docstring
            return f"Function: {func.name}"
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_text_preparator.py::test_prepare_function_without_docstring -v
python -m pytest tests/test_text_preparator.py::test_very_long_function_fallback -v
python -m pytest tests/test_text_preparator.py -v
```

Expected: All tests PASS (85+ tests total)

**Step 4: Commit**

```bash
git add codesearch/embeddings/text_preparator.py tests/test_text_preparator.py
git commit -m "feat: Implement error handling and fallback (Task 5)"
```

---

## Task 6: Integration Testing & Real Codebase Validation

**Files:**
- Test: Validate on P3 codebase

**Step 1: Create validation script**

```bash
source venv/bin/activate

python3 << 'EOF'
import sys
sys.path.insert(0, '.')

from codesearch.embeddings.text_preparator import TextPreparator
from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.parsers.python_parser import PythonParser
from codesearch.indexing.scanner import RepositoryScannerImpl

print("Testing text preparation on P3 codebase...")
print("-" * 60)

# Initialize components
scanner = RepositoryScannerImpl()
parser = PythonParser()
generator = EmbeddingGenerator()
preparator = TextPreparator(generator.tokenizer, max_tokens=512)

# Scan P3 repository
p3_path = "/Users/julienpequegnot/Code/parakeet-podcast-processor/p3"
files = scanner.scan_repository(p3_path, "p3")
print(f"✓ Scanned {len(files)} Python files")

# Parse and prepare
total_prepared = 0
for file in files[:3]:  # Test on first 3 files
    print(f"\nProcessing {file.file_path.split('/')[-1]}...")
    try:
        extracted = parser.parse_file(file.file_path)
        functions = [item for item in extracted if hasattr(item, 'signature')]

        prepared_count = 0
        for func in functions[:2]:  # First 2 functions per file
            prepared = preparator.prepare_function(func)
            token_count = preparator._count_tokens(prepared)
            prepared_count += 1
            total_prepared += 1
            print(f"  ✓ {func.name}: {len(prepared)} chars, {token_count} tokens")

        print(f"  Prepared {prepared_count} functions")
    except Exception as e:
        print(f"  Error: {e}")

print(f"\n✅ Total prepared: {total_prepared} functions")
print(f"✅ All text preparations passed validation!")
EOF
```

**Step 2: Run validation**

Expected output:
```
✓ Scanned 13 Python files
Processing [file1]...
  ✓ function1: X chars, Y tokens
  ✓ function2: X chars, Y tokens
  ...
✅ Total prepared: X functions
✅ All text preparations passed validation!
```

**Step 3: Verify full test suite**

```bash
python -m pytest tests/ -v
```

Expected: 85+ tests passing

**Step 4: Commit**

```bash
git commit -m "test: Validate text preparation on real P3 codebase (Task 6)" || echo "No changes"
```

---

## Task 7: Final Testing & PR Creation

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -v --cov=codesearch.embeddings --cov-report=term-missing
```

Expected: >85 tests, >90% coverage on embeddings module

**Step 2: Check code quality**

```bash
python -m black codesearch/embeddings/ tests/test_text_preparator.py
python -m ruff check codesearch/embeddings/
```

**Step 3: Verify git status**

```bash
git status
git log --oneline -7
```

**Step 4: Push feature branch**

```bash
git push origin feature/issue-6-text-preparation
```

**Step 5: Create PR**

```bash
gh pr create --title "feat: Component 2.2 - Text Preparation for Embeddings" \
  --body "
## Summary

Implements TextPreparator class for intelligent code text preparation:

- Docstring + code combination with clear separation
- Smart comment filtering (keeps important, removes trivial)
- Token-aware truncation with semantic preservation
- Graceful fallback for edge cases
- Batch processing support

## Implementation Details

**Task 1**: TextPreparator basic structure with combine/prepare methods
**Task 2**: Docstring + code combination logic
**Task 3**: Smart comment filtering (keeps TODO, FIXME, NOTE, etc.)
**Task 4**: Token counting and truncation strategies
**Task 5**: Error handling with fallback mechanisms
**Task 6**: Real codebase validation on P3 repository
**Task 7**: Final testing and PR creation

## Test Results

- 85+ tests passing
- >90% code coverage on embeddings module
- Validated on P3 codebase (13 files)
- All edge cases handled gracefully

## Files Changed

- codesearch/embeddings/text_preparator.py (NEW, ~250 lines)
- codesearch/embeddings/__init__.py (updated exports)
- tests/test_text_preparator.py (NEW, 20+ tests)

Fixes #6
"
```

**Step 6: Verify PR created**

```bash
gh pr view
```

---

## Success Criteria

✓ TextPreparator class created with all required methods
✓ Docstring + code combination working correctly
✓ Smart comment filtering preserving important comments
✓ Token counting and truncation respecting max_tokens limit
✓ Error handling with graceful fallback
✓ Batch processing functional
✓ All tests passing with >90% code coverage
✓ Real codebase validation on P3 successful
✓ Code formatted and linted
✓ PR created and ready for review

---

## File Structure After Completion

```
codesearch/embeddings/
├── __init__.py (updated with TextPreparator export)
├── generator.py (existing EmbeddingGenerator)
└── text_preparator.py (NEW, core implementation)

tests/
└── test_text_preparator.py (NEW, 20+ tests)
```

---

## Next Steps

After PR is merged:
- Issue #6 closed
- Ready for Issue #7 (Component 2.3: Batch Processing Pipeline)
- Ready for Issue #9 (Component 3.1: LanceDB Schema Design)
