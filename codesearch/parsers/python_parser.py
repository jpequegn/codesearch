"""Python code parser using the ast module."""

import ast
from typing import List, Set

from codesearch.models import Function
from codesearch.parsers.base import BaseParser


class PythonParser(BaseParser):
    """Parser for Python source code using the built-in ast module."""

    def get_language(self) -> str:
        """Return the language this parser handles."""
        return "python"

    def parse_file(self, file_path: str) -> List[Function]:
        """
        Parse a Python file and extract functions.

        Args:
            file_path: Path to the Python file

        Returns:
            List of Function objects extracted from the file

        Raises:
            SyntaxError: If the file contains invalid Python syntax
            IOError: If the file cannot be read
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except IOError as e:
            raise IOError(f"Cannot read file {file_path}: {e}")

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in {file_path}: {e}")

        functions = []
        source_lines = source.split("\n")

        # Extract top-level functions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func = self._extract_function(node, source_lines, file_path)
                functions.append(func)

        return functions

    def _extract_function(
        self, node: ast.FunctionDef, source_lines: List[str], file_path: str
    ) -> Function:
        """
        Extract metadata for a single function.

        Args:
            node: AST FunctionDef node
            source_lines: Source file split by lines
            file_path: Path to the source file

        Returns:
            Function object with extracted metadata
        """
        # Extract source code
        start_line = node.lineno - 1  # ast uses 1-based indexing
        end_line = node.end_lineno if node.end_lineno else len(source_lines)
        source_code = "\n".join(source_lines[start_line:end_line])

        # Extract signature
        signature = self._get_signature(node)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract function calls
        calls_to = self._get_calls(node)

        return Function(
            name=node.name,
            file_path=file_path,
            language="python",
            source_code=source_code,
            docstring=docstring,
            line_number=node.lineno,
            end_line=end_line,
            signature=signature,
            calls_to=list(calls_to),
            called_by=[],  # Populated in call graph phase
        )

    def _get_signature(self, node: ast.FunctionDef) -> str:
        """
        Extract function signature.

        Args:
            node: AST FunctionDef node

        Returns:
            Function signature string (e.g., "def foo(a: int, b: str) -> bool:")
        """
        args = node.args
        arg_strs = [arg.arg for arg in args.args]

        # Add type hints if present
        arg_strs_with_hints = []
        for i, arg in enumerate(args.args):
            if arg.annotation:
                arg_strs_with_hints.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
            else:
                arg_strs_with_hints.append(arg.arg)

        # Handle return type annotation
        return_type = ""
        if node.returns:
            return_type = f" -> {ast.unparse(node.returns)}"

        return f"def {node.name}({', '.join(arg_strs_with_hints)}){return_type}:"

    def _get_calls(self, node: ast.FunctionDef) -> Set[str]:
        """
        Extract functions called by this function.

        Args:
            node: AST FunctionDef node

        Returns:
            Set of function names called by this function
        """
        calls = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # Handle simple function calls: foo()
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                # Handle method calls: obj.method()
                elif isinstance(child.func, ast.Attribute):
                    # For now, skip method calls as they're harder to track
                    # Could be extended to track "obj.method_name"
                    pass

        return calls
