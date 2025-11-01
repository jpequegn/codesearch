"""Tests for call graph analysis."""

import tempfile
from pathlib import Path

import pytest

from codesearch.models import CallGraph, Function, Import
from codesearch.parsers.python_parser import PythonParser


@pytest.fixture
def parser():
    """Create a Python parser instance."""
    return PythonParser()


@pytest.fixture
def temp_module_a():
    """Create a temporary module A."""
    code = '''"""Module A."""

def helper_func():
    """Helper function."""
    return 42

def process_data(data):
    """Process data by calling helper."""
    result = helper_func()
    return result + data
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink()


@pytest.fixture
def temp_module_b():
    """Create a temporary module B that imports from A."""
    code = '''"""Module B."""

def helper_func():
    """Different helper in module B."""
    return 100

def combine(a, b):
    """Combine two values."""
    return a + b

def main():
    """Main function using helpers."""
    x = helper_func()
    y = combine(x, 50)
    return y
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink()


@pytest.fixture
def simple_call_graph():
    """Create a simple call graph for testing."""
    graph = CallGraph()

    # Add some functions
    func_a = Function(
        name="func_a",
        file_path="/test/file.py",
        line_number=1,
        calls_to=["func_b"],
    )
    func_b = Function(
        name="func_b",
        file_path="/test/file.py",
        line_number=10,
        calls_to=["func_c"],
    )
    func_c = Function(
        name="func_c",
        file_path="/test/file.py",
        line_number=20,
        calls_to=[],
    )

    graph.add_function(func_a)
    graph.add_function(func_b)
    graph.add_function(func_c)

    return graph


# ==================== Import Extraction Tests ====================


def test_extract_simple_import(parser, temp_module_a):
    """Test extraction of simple import statements."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write("import os\nimport sys\n")
        temp_path = f.name

    try:
        imports = parser.extract_imports(temp_path)
        assert len(imports) == 2
        assert imports[0].module == "os"
        assert imports[0].import_type == "import"
        assert imports[1].module == "sys"
    finally:
        Path(temp_path).unlink()


def test_extract_from_import(parser):
    """Test extraction of from...import statements."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write("from pathlib import Path\nfrom os.path import join\n")
        temp_path = f.name

    try:
        imports = parser.extract_imports(temp_path)
        assert len(imports) == 2
        assert imports[0].module == "pathlib"
        assert imports[0].import_type == "from"
        assert "Path" in imports[0].names
        assert imports[1].module == "os.path"
        assert "join" in imports[1].names
    finally:
        Path(temp_path).unlink()


def test_extract_aliased_imports(parser):
    """Test extraction of aliased imports."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write("import numpy as np\nfrom os import path as p\n")
        temp_path = f.name

    try:
        imports = parser.extract_imports(temp_path)
        assert len(imports) == 2
        assert imports[0].names["numpy"] == "np"
        assert imports[1].names["path"] == "p"
    finally:
        Path(temp_path).unlink()


def test_extract_multiple_imports(parser):
    """Test extraction of multiple items in one import."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write("from collections import deque, defaultdict, Counter\n")
        temp_path = f.name

    try:
        imports = parser.extract_imports(temp_path)
        assert len(imports) == 1
        assert len(imports[0].names) == 3
        assert "deque" in imports[0].names
        assert "defaultdict" in imports[0].names
        assert "Counter" in imports[0].names
    finally:
        Path(temp_path).unlink()


def test_import_get_original_name():
    """Test Import.get_original_name method."""
    imp = Import(
        module="os",
        names={"path": None, "environ": None},
        import_type="from",
        file_path="/test/file.py",
        line_number=1,
    )

    assert imp.get_original_name("path") == "os.path"
    assert imp.get_original_name("environ") == "os.environ"
    assert imp.get_original_name("nonexistent") is None


def test_import_get_original_name_aliased():
    """Test Import.get_original_name with aliases."""
    imp = Import(
        module="collections",
        names={"defaultdict": "dd"},
        import_type="from",
        file_path="/test/file.py",
        line_number=1,
    )

    assert imp.get_original_name("dd") == "collections.defaultdict"


# ==================== Call Graph Tests ====================


def test_call_graph_add_function(simple_call_graph):
    """Test adding functions to call graph."""
    assert len(simple_call_graph.functions) == 3
    assert "/test/file.py:1" in simple_call_graph.functions


def test_call_graph_resolve_call_local(simple_call_graph):
    """Test resolving a call within the same file."""
    imports = []
    result = simple_call_graph.resolve_call("/test/file.py", "func_b", imports)

    assert result is not None
    func_key, func = result
    assert func.name == "func_b"


def test_call_graph_resolve_call_not_found(simple_call_graph):
    """Test resolution fails for non-existent function."""
    imports = []
    result = simple_call_graph.resolve_call("/test/file.py", "nonexistent", imports)
    assert result is None


def test_call_graph_build_called_by(simple_call_graph):
    """Test building called_by relationships."""
    simple_call_graph.build_called_by()

    # func_a calls func_b, so func_b.called_by should contain func_a's key
    func_b = simple_call_graph.functions["/test/file.py:10"]
    assert "/test/file.py:1" in func_b.called_by

    # func_b calls func_c
    func_c = simple_call_graph.functions["/test/file.py:20"]
    assert "/test/file.py:10" in func_c.called_by

    # func_c is not called by anyone
    func_a = simple_call_graph.functions["/test/file.py:1"]
    assert len(func_a.called_by) == 0


def test_call_graph_get_call_chain(simple_call_graph):
    """Test getting transitive call chain."""
    simple_call_graph.build_called_by()

    # func_a calls func_b and func_c (transitively)
    chain = simple_call_graph.get_call_chain("/test/file.py:1")
    assert "/test/file.py:10" in chain  # func_b
    assert "/test/file.py:20" in chain  # func_c

    # func_b only calls func_c
    chain_b = simple_call_graph.get_call_chain("/test/file.py:10")
    assert "/test/file.py:20" in chain_b
    assert "/test/file.py:1" not in chain_b

    # func_c doesn't call anything
    chain_c = simple_call_graph.get_call_chain("/test/file.py:20")
    assert len(chain_c) == 0


def test_call_graph_get_callers(simple_call_graph):
    """Test getting direct callers."""
    simple_call_graph.build_called_by()

    # func_b is called by func_a
    callers_b = simple_call_graph.get_callers("/test/file.py:10")
    assert "/test/file.py:1" in callers_b

    # func_c is called by func_b
    callers_c = simple_call_graph.get_callers("/test/file.py:20")
    assert "/test/file.py:10" in callers_c

    # func_a is not called by anyone
    callers_a = simple_call_graph.get_callers("/test/file.py:1")
    assert len(callers_a) == 0


# ==================== Integration Tests ====================


def test_parser_extract_imports_and_functions(parser, temp_module_a):
    """Test extracting both functions and imports from a file."""
    extracted = parser.parse_file(temp_module_a)
    imports = parser.extract_imports(temp_module_a)

    # Check functions extracted
    from codesearch.models import Function

    functions = [item for item in extracted if isinstance(item, Function)]
    assert len(functions) == 2
    assert any(f.name == "helper_func" for f in functions)
    assert any(f.name == "process_data" for f in functions)

    # Check imports
    assert len(imports) == 0  # No imports in this file


def test_parser_build_call_graph(parser, temp_module_a):
    """Test building a call graph from parsed file."""
    extracted = parser.parse_file(temp_module_a)
    imports = parser.extract_imports(temp_module_a)

    from codesearch.models import Function

    functions = [item for item in extracted if isinstance(item, Function)]

    # Build extracted_by_file format
    extracted_by_file = {temp_module_a: (functions, [], imports)}

    graph, unresolved = parser.build_call_graph(extracted_by_file)

    # Check graph built
    assert len(graph.functions) == 2

    # Check process_data calls helper_func
    process_data = next(f for f in functions if f.name == "process_data")
    helper_func = next(f for f in functions if f.name == "helper_func")

    assert "helper_func" in process_data.calls_to

    # Resolve calls
    graph.build_called_by()
    assert len(helper_func.called_by) > 0


# ==================== Edge Cases ====================


def test_call_graph_circular_calls():
    """Test handling of circular function calls."""
    graph = CallGraph()

    # Create circular calls: a -> b -> c -> a
    func_a = Function(
        name="func_a", file_path="/test/file.py", line_number=1, calls_to=["func_b"]
    )
    func_b = Function(
        name="func_b", file_path="/test/file.py", line_number=10, calls_to=["func_c"]
    )
    func_c = Function(
        name="func_c", file_path="/test/file.py", line_number=20, calls_to=["func_a"]
    )

    graph.add_function(func_a)
    graph.add_function(func_b)
    graph.add_function(func_c)

    # Get call chain - should not infinite loop
    chain = graph.get_call_chain("/test/file.py:1")
    assert "/test/file.py:10" in chain
    assert "/test/file.py:20" in chain
    assert len(chain) == 2  # Only 2 others, not infinite


def test_import_extraction_with_no_imports(parser):
    """Test extraction from file with no imports."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write("def func():\n    return 42\n")
        temp_path = f.name

    try:
        imports = parser.extract_imports(temp_path)
        assert len(imports) == 0
    finally:
        Path(temp_path).unlink()


def test_call_graph_unresolved_external_calls(parser, temp_module_a):
    """Test tracking of unresolved external calls."""
    extracted = parser.parse_file(temp_module_a)
    imports = parser.extract_imports(temp_module_a)

    from codesearch.models import Function

    functions = [item for item in extracted if isinstance(item, Function)]

    # Create a function with external call (like json.dumps)
    external_call = Function(
        name="to_json",
        file_path=temp_module_a,
        line_number=50,
        calls_to=["dumps"],  # json.dumps but json not imported
    )
    functions.append(external_call)

    extracted_by_file = {temp_module_a: (functions, [], imports)}
    graph, unresolved = parser.build_call_graph(extracted_by_file)

    # Check unresolved calls tracked
    assert len(unresolved[temp_module_a]) > 0
    assert any(call[1] == "dumps" for call in unresolved[temp_module_a])
