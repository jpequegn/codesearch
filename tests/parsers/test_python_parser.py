"""Unit tests for Python parser."""

import tempfile
from pathlib import Path

import pytest

from codesearch.parsers.python_parser import PythonParser
from codesearch.models import Function, Class


class TestPythonParserBasic:
    """Basic functionality tests for Python parser."""

    def test_parser_language(self):
        """Test that parser returns correct language."""
        parser = PythonParser()
        assert parser.get_language() == "python"

    def test_parse_simple_function(self):
        """Test parsing a simple function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def hello_world():
    '''A simple greeting function.'''
    print("Hello, World!")
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            assert len(entities) > 0
            assert any(e.name == "hello_world" for e in entities if isinstance(e, Function))

    def test_parse_function_with_parameters(self):
        """Test parsing function with parameters."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def add(a, b):
    '''Add two numbers.'''
    return a + b
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            functions = [e for e in entities if isinstance(e, Function)]
            assert len(functions) > 0

            add_func = next((f for f in functions if f.name == "add"), None)
            assert add_func is not None

    def test_parse_simple_class(self):
        """Test parsing a simple class."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
class Dog:
    '''A dog class.'''
    def __init__(self, name):
        self.name = name

    def bark(self):
        return f"{self.name} says woof!"
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            assert len(entities) > 0
            assert any(e.name == "Dog" for e in entities if isinstance(e, Class))

    def test_parse_nested_functions(self):
        """Test parsing nested functions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def outer():
    def inner():
        return "nested"
    return inner()
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            # Should find both outer and inner functions
            functions = [e for e in entities if isinstance(e, Function)]
            assert any(f.name == "outer" for f in functions)

    def test_parse_async_function(self):
        """Test parsing async function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
async def fetch_data():
    '''Fetch data asynchronously.'''
    return "data"
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            functions = [e for e in entities if isinstance(e, Function)]
            assert any(f.is_async for f in functions if f.name == "fetch_data")

    def test_parse_class_with_methods(self):
        """Test parsing class with multiple methods."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    @staticmethod
    def multiply(a, b):
        return a * b
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            functions = [e for e in entities if isinstance(e, Function) and e.is_method]
            assert len(functions) >= 2  # At least add and subtract

    def test_parse_file_not_found(self):
        """Test handling of missing file."""
        parser = PythonParser()
        with pytest.raises(IOError):
            parser.parse_file("/nonexistent/file.py")

    def test_parse_syntax_error(self):
        """Test handling of syntax errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def broken(:\n")
            f.flush()

            parser = PythonParser()
            with pytest.raises(SyntaxError):
                parser.parse_file(f.name)

    def test_parse_with_imports(self):
        """Test parsing file with imports."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import os
from typing import List

def process_files(paths: List[str]):
    for path in paths:
        print(path)
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            functions = [e for e in entities if isinstance(e, Function)]
            assert any(f.name == "process_files" for f in functions)

    def test_parse_with_decorators(self):
        """Test parsing functions with decorators."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def decorated_function():
    pass
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            functions = [e for e in entities if isinstance(e, Function)]
            assert any(f.name == "decorated_function" for f in functions)

    def test_parse_multiple_classes(self):
        """Test parsing file with multiple classes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
class Animal:
    pass

class Dog(Animal):
    pass

class Cat(Animal):
    pass
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            classes = [e for e in entities if isinstance(e, Class)]
            assert len(classes) >= 3

    def test_function_metadata(self):
        """Test that function metadata is correctly extracted."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def example_func(a, b=10):
    '''Example function with docstring.'''
    return a + b
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            func = next((e for e in entities if isinstance(e, Function) and e.name == "example_func"), None)
            assert func is not None
            assert func.source_code is not None
            assert len(func.source_code) > 0
            assert func.docstring is not None

    def test_class_metadata(self):
        """Test that class metadata is correctly extracted."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
class MyClass:
    '''A sample class.'''
    pass
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            cls = next((e for e in entities if isinstance(e, Class)), None)
            assert cls is not None
            assert cls.source_code is not None
            assert cls.docstring is not None

    def test_parse_empty_file(self):
        """Test parsing empty Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            assert len(entities) == 0

    def test_parse_comments_only(self):
        """Test parsing file with only comments."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
# This is a comment
# Another comment
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            assert len(entities) == 0

    def test_parse_complex_structure(self):
        """Test parsing complex file with multiple entities."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def module_function():
    pass

class FirstClass:
    def method1(self):
        pass

    def method2(self):
        pass

class SecondClass:
    pass

def another_function():
    pass
""")
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            # Should find all entities
            assert len(entities) > 0

            functions = [e for e in entities if isinstance(e, Function)]
            classes = [e for e in entities if isinstance(e, Class)]

            assert len(classes) >= 2
            assert len(functions) >= 2

    def test_function_source_code(self):
        """Test that source code is correctly extracted."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            code = """def my_func():
    x = 1
    return x"""
            f.write(code)
            f.flush()

            parser = PythonParser()
            entities = parser.parse_file(f.name)

            func = next((e for e in entities if isinstance(e, Function)), None)
            assert func is not None
            assert "my_func" in func.source_code
            assert "x = 1" in func.source_code
