"""Tests for Python parser."""

import tempfile
from pathlib import Path

import pytest

from codesearch.models import Class, Function
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
    """A test class with methods."""

    def method(self):
        """Class method."""
        pass

    def another_method(self):
        """Another method."""
        return self.method()
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def temp_class_file():
    """Create a temporary Python file with classes for testing."""
    code = '''"""Test file with various class patterns."""

class BaseClass:
    """Base class for testing."""

    def __init__(self, name: str):
        """Initialize."""
        self.name = name

    def get_name(self) -> str:
        """Get name."""
        return self.name


class DerivedClass(BaseClass):
    """Derived class inheriting from BaseClass."""

    def __init__(self, name: str, age: int):
        """Initialize with name and age."""
        super().__init__(name)
        self.age = age

    def get_info(self) -> str:
        """Get info string."""
        return f"{self.get_name()} is {self.age}"


class MultipleInheritance(BaseClass, dict):
    """Class with multiple inheritance."""
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


@pytest.fixture
def temp_nested_file():
    """Create a temporary Python file with nested functions."""
    code = '''"""Test file with nested functions."""

def outer_function():
    """Outer function with nested function."""

    def inner_function():
        """Inner function."""
        return "inner"

    def another_inner():
        """Another inner function."""
        return inner_function()

    return another_inner()


def with_closure():
    """Function with closure."""
    x = 10

    def closure_func():
        """Function using closure."""
        return x + 5

    return closure_func()


async def async_function():
    """Async function."""
    pass


async def async_with_nested():
    """Async function with nested async."""

    async def nested_async():
        """Nested async function."""
        pass

    await nested_async()
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
    extracted = parser.parse_file(temp_python_file)

    # Should find 4 top-level functions, 1 class, and 2 methods = 7 total
    from codesearch.models import Class
    functions = [item for item in extracted if isinstance(item, Function)]
    classes = [item for item in extracted if isinstance(item, Class)]

    assert len(functions) == 6  # 4 top-level + 2 methods
    assert len(classes) == 1

    # Check function names (including methods)
    top_level_names = {f.name for f in functions if not f.is_method}
    assert top_level_names == {
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


# ==================== Tests for Classes and Methods ====================


def test_parse_file_with_classes(parser, temp_python_file):
    """Test parsing a file with classes."""
    extracted = parser.parse_file(temp_python_file)

    # Should find: 4 functions + 1 class + 2 methods
    classes = [item for item in extracted if isinstance(item, Class)]
    functions = [item for item in extracted if isinstance(item, Function)]

    assert len(classes) == 1
    assert classes[0].name == "MyClass"
    assert classes[0].docstring == "A test class with methods."

    # Check methods
    methods = [f for f in functions if f.is_method]
    assert len(methods) == 2
    method_names = {m.name for m in methods}
    assert method_names == {"method", "another_method"}


def test_method_has_class_context(parser, temp_python_file):
    """Test that methods have class context."""
    extracted = parser.parse_file(temp_python_file)
    functions = [item for item in extracted if isinstance(item, Function)]
    methods = [f for f in functions if f.is_method]

    for method in methods:
        assert method.class_name == "MyClass"
        assert method.is_method is True
        assert method.fully_qualified_name() == f"MyClass.{method.name}"


def test_class_with_inheritance(parser, temp_class_file):
    """Test parsing classes with inheritance."""
    extracted = parser.parse_file(temp_class_file)
    classes = [item for item in extracted if isinstance(item, Class)]

    assert len(classes) == 3

    # Check base class
    base = next(c for c in classes if c.name == "BaseClass")
    assert base.bases == []

    # Check derived class
    derived = next(c for c in classes if c.name == "DerivedClass")
    assert "BaseClass" in derived.bases

    # Check multiple inheritance
    multi = next(c for c in classes if c.name == "MultipleInheritance")
    assert "BaseClass" in multi.bases
    assert "dict" in multi.bases


def test_method_extraction_from_class(parser, temp_class_file):
    """Test extraction of methods from classes."""
    extracted = parser.parse_file(temp_class_file)
    functions = [item for item in extracted if isinstance(item, Function)]

    methods = [f for f in functions if f.is_method]
    # 2 methods from BaseClass + 2 from DerivedClass = 4
    # (MultipleInheritance has no methods)
    assert len(methods) >= 4


# ==================== Tests for Nested Functions ====================


def test_nested_function_extraction(parser, temp_nested_file):
    """Test extraction of nested functions."""
    extracted = parser.parse_file(temp_nested_file)
    functions = [item for item in extracted if isinstance(item, Function)]

    # Should have: outer_function, inner_function, another_inner, with_closure, closure_func, async_function, async_with_nested, nested_async
    assert len(functions) >= 6

    # Check outer function
    outer = next(f for f in functions if f.name == "outer_function" and f.depth == 0)
    assert outer.depth == 0

    # Check nested functions
    inner = next(f for f in functions if f.name == "inner_function")
    assert inner.depth == 1
    assert inner.parent_function == "outer_function"
    assert inner.fully_qualified_name() == "outer_function.inner_function"

    # Check another nested
    another = next(f for f in functions if f.name == "another_inner")
    assert another.depth == 1
    assert another.parent_function == "outer_function"


def test_closure_detection(parser, temp_nested_file):
    """Test detection of closures."""
    extracted = parser.parse_file(temp_nested_file)
    functions = [item for item in extracted if isinstance(item, Function)]

    closure = next(f for f in functions if f.name == "closure_func")
    assert closure.depth == 1
    assert closure.parent_function == "with_closure"


def test_async_function_detection(parser, temp_nested_file):
    """Test detection of async functions."""
    extracted = parser.parse_file(temp_nested_file)
    functions = [item for item in extracted if isinstance(item, Function)]

    async_func = next(f for f in functions if f.name == "async_function")
    assert async_func.is_async is True
    assert "async def" in async_func.signature

    # Check nested async
    nested_async = next(f for f in functions if f.name == "nested_async")
    assert nested_async.is_async is True
    assert nested_async.parent_function == "async_with_nested"


# ==================== Tests for Fully Qualified Names ====================


def test_fully_qualified_names(parser, temp_class_file):
    """Test fully qualified name generation."""
    extracted = parser.parse_file(temp_class_file)
    functions = [item for item in extracted if isinstance(item, Function)]

    # Method
    init = next(f for f in functions if f.name == "__init__" and f.class_name == "BaseClass")
    assert init.fully_qualified_name() == "BaseClass.__init__"

    # Nested function
    extracted2 = parser.parse_file(temp_class_file)
    # This would need a file with nested functions


# ==================== Edge Cases ====================


def test_class_with_only_docstring(parser):
    """Test parsing a class with only a docstring."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            '''class EmptyClass:
    """An empty class."""
    pass
'''
        )
        temp_path = f.name

    try:
        extracted = parser.parse_file(temp_path)
        classes = [item for item in extracted if isinstance(item, Class)]
        assert len(classes) == 1
        assert classes[0].name == "EmptyClass"
    finally:
        Path(temp_path).unlink()
