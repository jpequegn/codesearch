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


def test_count_tokens_simple(tokenizer):
    """Test token counting functionality."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    code = "def hello(): return 'world'"
    count = preparator._count_tokens(code)

    # Should be positive integer
    assert isinstance(count, int)
    assert count > 0
    assert count < 512  # Under max


def test_truncate_under_limit(tokenizer):
    """Test that text under token limit is not truncated."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    code = "def simple(): return 42"
    truncated = preparator._truncate_to_tokens(code)

    # Should be identical if under limit
    assert truncated == code


def test_truncate_over_limit(tokenizer):
    """Test that text over token limit gets truncated intelligently."""
    preparator = TextPreparator(tokenizer, max_tokens=100)  # Very small limit

    # Create code that exceeds token limit
    long_code = "def process():\n    " + "\n    ".join([f"x{i} = {i}" for i in range(50)])

    truncated = preparator._truncate_to_tokens(long_code)

    # Should be truncated
    assert len(truncated) < len(long_code)
    # Token count should respect limit
    token_count = preparator._count_tokens(truncated)
    assert token_count <= 100


def test_truncate_preserves_docstring(tokenizer):
    """Test that truncation always preserves docstring if present."""
    preparator = TextPreparator(tokenizer, max_tokens=50)  # Tight limit

    # Create function with docstring and long code
    long_code = "def f():\n    " + "\n    ".join([f"x{i} = {i}" for i in range(100)])
    docstring = "Important documentation for this function."

    func = Function(
        name="f",
        file_path="/test.py",
        language="python",
        source_code=long_code,
        docstring=docstring,
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Docstring should be preserved in truncated output
    assert docstring in prepared
    # Should respect token limit
    assert preparator._count_tokens(prepared) <= 50


def test_prepare_long_function_truncation(tokenizer):
    """Test complete preparation with truncation of long function."""
    preparator = TextPreparator(tokenizer, max_tokens=100)

    # Create very long function
    long_code = '''def analyze_data():
    """Analyze and process large dataset."""
    # TODO: optimize performance
    data = load_data()
    ''' + "\n    ".join([f"result{i} = process(data[{i}])" for i in range(50)])

    func = Function(
        name="analyze_data",
        file_path="/test.py",
        language="python",
        source_code=long_code,
        docstring="Analyze and process large dataset.",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Should have important elements
    assert "Analyze and process" in prepared or "analyze_data" in prepared
    # Must respect token limit
    token_count = preparator._count_tokens(prepared)
    assert token_count <= 100


def test_prepare_function_empty_code(tokenizer):
    """Test preparing function with empty source code."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    func = Function(
        name="empty_func",
        file_path="/test.py",
        language="python",
        source_code="",
        docstring="Function with no implementation.",
        line_number=1,
    )

    # Should not raise, should return graceful fallback
    prepared = preparator.prepare_function(func)
    assert isinstance(prepared, str)
    assert len(prepared) > 0


def test_prepare_function_no_docstring(tokenizer):
    """Test preparing function with no docstring."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    func = Function(
        name="nodoc",
        file_path="/test.py",
        language="python",
        source_code="def nodoc(): pass",
        docstring=None,
        line_number=1,
    )

    prepared = preparator.prepare_function(func)
    assert "def nodoc" in prepared


def test_prepare_function_empty_docstring(tokenizer):
    """Test preparing function with empty string docstring."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    func = Function(
        name="empty_doc",
        file_path="/test.py",
        language="python",
        source_code="def empty_doc(): return 1",
        docstring="",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)
    assert "def empty_doc" in prepared


def test_prepare_class_with_no_docstring(tokenizer):
    """Test preparing class with no docstring."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    cls = Class(
        name="NoDocClass",
        file_path="/test.py",
        language="python",
        source_code="class NoDocClass:\n    pass",
        docstring=None,
        line_number=1,
    )

    prepared = preparator.prepare_class(cls)
    assert "NoDocClass" in prepared


def test_prepare_batch_with_empty_list(tokenizer):
    """Test batch preparation with empty list."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    result = preparator.prepare_batch([])

    assert isinstance(result, list)
    assert len(result) == 0


def test_prepare_batch_mixed_items(tokenizer):
    """Test batch preparation with both functions and classes."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    func = Function(
        name="func1",
        file_path="/test.py",
        language="python",
        source_code="def func1(): pass",
        docstring="A function.",
        line_number=1,
    )

    cls = Class(
        name="Class1",
        file_path="/test.py",
        language="python",
        source_code="class Class1: pass",
        docstring="A class.",
        line_number=5,
    )

    results = preparator.prepare_batch([func, cls])

    assert len(results) == 2
    assert "func1" in results[0]
    assert "Class1" in results[1]


def test_prepare_function_with_unicode(tokenizer):
    """Test preparing function with unicode characters."""
    preparator = TextPreparator(tokenizer, max_tokens=512)

    func = Function(
        name="unicode_func",
        file_path="/test.py",
        language="python",
        source_code='def unicode_func():\n    # 日本語コメント\n    return "你好"',
        docstring="Unicode aware function. 🚀",
        line_number=1,
    )

    prepared = preparator.prepare_function(func)

    # Should not raise, should handle unicode gracefully
    assert isinstance(prepared, str)
    assert len(prepared) > 0
