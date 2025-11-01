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


def test_combine_docstring_and_code(tokenizer):
    """Test combining docstring and source code."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

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


def test_combine_no_docstring(tokenizer):
    """Test that code-only functions work."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

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


def test_combine_empty_docstring(tokenizer):
    """Test empty docstring handling."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

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


def test_prepare_class(tokenizer):
    """Test preparing a class for embedding."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

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


def test_prepare_class_no_docstring(tokenizer):
    """Test preparing a class without docstring."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    cls_code = "class Empty:\n    pass"

    cls = Class(
        name="Empty",
        file_path="/test.py",
        language="python",
        source_code=cls_code,
        docstring=None,
        line_number=1,
    )

    prepared = preparator.prepare_class(cls)

    # Should just be the code
    assert prepared == cls_code
    assert "class Empty" in prepared


def test_separator_in_combined_text(tokenizer):
    """Test that separator is present when both docstring and code exist."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    func = Function(
        name="test",
        file_path="/test.py",
        language="python",
        source_code="def test(): return 1",
        docstring="Test function.",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Check separator exists
    assert "\n\n" in prepared
    # Check order: docstring first, then code
    parts = prepared.split("\n\n")
    assert len(parts) == 2
    assert parts[0] == "Test function."
    assert "def test" in parts[1]


def test_filter_important_comments(tokenizer):
    """Test that important comments (TODO, FIXME, etc.) are kept."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    code_with_comments = '''def process_data(data):
    # TODO: add error handling
    result = []
    for item in data:
        # FIXME: optimize loop
        if item > 0:
            result.append(item)
    return result'''

    filtered = preparator._filter_comments(code_with_comments)

    # Important comments should be preserved
    assert "TODO" in filtered
    assert "FIXME" in filtered


def test_filter_trivial_comments(tokenizer):
    """Test that trivial comments without keywords are removed."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    code_with_trivial = '''def add(a, b):
    # add two numbers
    x = a + b  # calculate sum
    return x  # return result'''

    filtered = preparator._filter_comments(code_with_trivial)

    # Trivial comments should be removed
    assert "add two numbers" not in filtered
    assert "calculate sum" not in filtered
    assert "return result" not in filtered
    # But structure should be preserved
    assert "def add" in filtered
    assert "return x" in filtered


def test_keep_block_comments(tokenizer):
    """Test that block comments are preserved."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    code_with_block = '''def complex_algorithm():
    # This is a multi-line block comment
    # explaining the algorithm approach
    # and implementation details
    result = calculate()
    return result'''

    filtered = preparator._filter_comments(code_with_block)

    # Block comments should be preserved
    assert "multi-line block comment" in filtered
    assert "algorithm approach" in filtered


def test_prepare_with_comment_filtering(tokenizer):
    """Test that prepare_function applies comment filtering."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    func = Function(
        name="process",
        file_path="/test.py",
        language="python",
        source_code='''def process(x):
    # important calculation
    # TODO: add validation
    result = x * 2  # double it
    return result''',
        docstring="Process input value.",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Important comments preserved
    assert "TODO" in prepared
    # Trivial comments removed
    assert "double it" not in prepared
