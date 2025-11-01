"""Tests for Python parser."""

import tempfile
from pathlib import Path

import pytest

from codesearch.models import Function
from codesearch.parsers.python_parser import PythonParser


@pytest.fixture
def parser():
    """Create a Python parser instance."""
    return PythonParser()


@pytest.fixture
def temp_python_file():
    """Create a temporary Python file for testing."""
    code = '''"""Module docstring."""

def simple_function():
    """Simple function with docstring."""
    pass


def function_with_args(a: int, b: str) -> bool:
    """Function with typed arguments and return type."""
    return True


def function_with_calls():
    """Function that calls other functions."""
    x = simple_function()
    y = function_with_args(1, "test")
    return x and y


def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


class MyClass:
    """A test class - methods will be skipped in this version."""

    def method(self):
        """Class method."""
        pass
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


def test_parser_get_language(parser):
    """Test that parser identifies itself as Python."""
    assert parser.get_language() == "python"


def test_parse_file_basic(parser, temp_python_file):
    """Test basic parsing of a Python file."""
    functions = parser.parse_file(temp_python_file)

    # Should find 4 top-level functions (class methods excluded)
    assert len(functions) == 4

    # Check function names
    names = {f.name for f in functions}
    assert names == {
        "simple_function",
        "function_with_args",
        "function_with_calls",
        "factorial",
    }


def test_parse_file_not_found(parser):
    """Test that parser raises error for non-existent file."""
    with pytest.raises(IOError):
        parser.parse_file("/nonexistent/file.py")


def test_parse_syntax_error(parser):
    """Test that parser raises error for invalid Python syntax."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write("def invalid syntax here")
        temp_path = f.name

    try:
        with pytest.raises(SyntaxError):
            parser.parse_file(temp_path)
    finally:
        Path(temp_path).unlink()


def test_function_metadata_simple(parser, temp_python_file):
    """Test extraction of simple function metadata."""
    functions = parser.parse_file(temp_python_file)
    simple = next(f for f in functions if f.name == "simple_function")

    assert simple.name == "simple_function"
    assert simple.file_path == temp_python_file
    assert simple.language == "python"
    assert simple.docstring == "Simple function with docstring."
    assert "def simple_function():" in simple.signature
    assert simple.line_number > 0
    assert simple.end_line > simple.line_number


def test_function_with_typed_args(parser, temp_python_file):
    """Test extraction of function signature with type hints."""
    functions = parser.parse_file(temp_python_file)
    typed_func = next(f for f in functions if f.name == "function_with_args")

    assert "a: int" in typed_func.signature
    assert "b: str" in typed_func.signature
    assert "-> bool" in typed_func.signature


def test_function_calls_extraction(parser, temp_python_file):
    """Test extraction of function calls."""
    functions = parser.parse_file(temp_python_file)
    caller = next(f for f in functions if f.name == "function_with_calls")

    # Should identify calls to other functions
    assert "simple_function" in caller.calls_to
    assert "function_with_args" in caller.calls_to


def test_recursive_calls(parser, temp_python_file):
    """Test extraction of recursive calls."""
    functions = parser.parse_file(temp_python_file)
    factorial = next(f for f in functions if f.name == "factorial")

    # factorial calls itself
    assert "factorial" in factorial.calls_to


def test_source_code_extraction(parser, temp_python_file):
    """Test that source code is correctly extracted."""
    functions = parser.parse_file(temp_python_file)
    simple = next(f for f in functions if f.name == "simple_function")

    assert "def simple_function():" in simple.source_code
    assert "pass" in simple.source_code


def test_function_equality():
    """Test Function equality based on identity."""
    func1 = Function(name="foo", file_path="/path/to/file.py", line_number=10)
    func2 = Function(name="foo", file_path="/path/to/file.py", line_number=10)
    func3 = Function(name="foo", file_path="/path/to/file.py", line_number=20)

    assert func1 == func2
    assert func1 != func3


def test_function_hashable():
    """Test that Function objects can be used in sets."""
    func1 = Function(name="foo", file_path="/path/to/file.py", line_number=10)
    func2 = Function(name="foo", file_path="/path/to/file.py", line_number=10)
    func3 = Function(name="bar", file_path="/path/to/file.py", line_number=20)

    function_set = {func1, func2, func3}
    assert len(function_set) == 2  # func1 and func2 are the same


def test_parse_empty_file(parser):
    """Test parsing an empty Python file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write("")
        temp_path = f.name

    try:
        functions = parser.parse_file(temp_path)
        assert len(functions) == 0
    finally:
        Path(temp_path).unlink()


def test_parse_file_with_only_imports(parser):
    """Test parsing a file with only imports and no functions."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write("import os\nimport sys\nfrom pathlib import Path\n")
        temp_path = f.name

    try:
        functions = parser.parse_file(temp_path)
        assert len(functions) == 0
    finally:
        Path(temp_path).unlink()
