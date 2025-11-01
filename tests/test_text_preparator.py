"""Tests for text preparation for embeddings."""

import pytest
from transformers import AutoTokenizer

from codesearch.embeddings.text_preparator import TextPreparator
from codesearch.models import Function, Class


@pytest.fixture
def tokenizer():
    """Fixture providing a tokenizer for testing."""
    return AutoTokenizer.from_pretrained("microsoft/codebert-base")


def test_text_preparator_initialization(tokenizer):
    """Test initializing TextPreparator with tokenizer."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    assert preparator is not None
    assert preparator.tokenizer is not None
    assert preparator.max_tokens == 512


def test_text_preparator_custom_max_tokens(tokenizer):
    """Test initializing TextPreparator with custom max_tokens."""
    preparator = TextPreparator(tokenizer, max_tokens=256)

    assert preparator.max_tokens == 256


def test_prepare_simple_function(tokenizer):
    """Test preparing a simple function."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

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


def test_prepare_function_without_docstring(tokenizer):
    """Test preparing a function without a docstring."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    func = Function(
        name="add",
        file_path="/math.py",
        language="python",
        source_code="def add(a, b):\n    return a + b",
        docstring=None,
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Should contain source code
    assert "def add" in prepared
    assert prepared == "def add(a, b):\n    return a + b"


def test_prepare_simple_class(tokenizer):
    """Test preparing a simple class."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    cls = Class(
        name="MyClass",
        file_path="/test.py",
        language="python",
        source_code="class MyClass:\n    pass",
        docstring="A test class.",
        line_number=1,
    )

    prepared = preparator.prepare_class(cls)

    # Should contain both docstring and source code
    assert "A test class" in prepared
    assert "class MyClass" in prepared
    assert isinstance(prepared, str)
    assert len(prepared) > 0


def test_prepare_class_without_docstring(tokenizer):
    """Test preparing a class without a docstring."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    cls = Class(
        name="SimpleClass",
        file_path="/test.py",
        language="python",
        source_code="class SimpleClass:\n    def __init__(self):\n        pass",
        docstring=None,
        line_number=1,
    )

    prepared = preparator.prepare_class(cls)

    # Should contain source code
    assert "class SimpleClass" in prepared
    assert prepared == "class SimpleClass:\n    def __init__(self):\n        pass"


def test_prepare_batch_mixed_items(tokenizer):
    """Test preparing a batch of mixed Function and Class items."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    func = Function(
        name="test_func",
        file_path="/test.py",
        language="python",
        source_code="def test_func():\n    pass",
        docstring="Test function.",
        line_number=1,
    )

    cls = Class(
        name="TestClass",
        file_path="/test.py",
        language="python",
        source_code="class TestClass:\n    pass",
        docstring="Test class.",
        line_number=5,
    )

    batch = [func, cls]
    prepared = preparator.prepare_batch(batch)

    assert len(prepared) == 2
    assert "Test function" in prepared[0]
    assert "def test_func" in prepared[0]
    assert "Test class" in prepared[1]
    assert "class TestClass" in prepared[1]


def test_prepare_batch_functions_only(tokenizer):
    """Test preparing a batch of functions only."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    funcs = [
        Function(
            name="func1",
            file_path="/test.py",
            language="python",
            source_code="def func1():\n    pass",
            docstring="First function.",
            line_number=1,
        ),
        Function(
            name="func2",
            file_path="/test.py",
            language="python",
            source_code="def func2():\n    pass",
            docstring="Second function.",
            line_number=5,
        ),
    ]

    prepared = preparator.prepare_batch(funcs)

    assert len(prepared) == 2
    assert "First function" in prepared[0]
    assert "Second function" in prepared[1]


def test_combine_text_with_docstring(tokenizer):
    """Test _combine_text method with docstring."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    result = preparator._combine_text("Test docstring.", "def func():\n    pass")

    assert result == "Test docstring.\n\ndef func():\n    pass"


def test_combine_text_without_docstring(tokenizer):
    """Test _combine_text method without docstring."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    result = preparator._combine_text(None, "def func():\n    pass")

    assert result == "def func():\n    pass"


def test_combine_text_empty_docstring(tokenizer):
    """Test _combine_text method with empty docstring."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    result = preparator._combine_text("", "def func():\n    pass")

    # Empty string is falsy, so should just return source code
    assert result == "def func():\n    pass"
