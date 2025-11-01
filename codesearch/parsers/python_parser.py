"""Python code parser using the ast module."""

import ast
from typing import List, Optional, Set, Union

from codesearch.models import Class, Function
from codesearch.parsers.base import BaseParser


class PythonParser(BaseParser):
    """Parser for Python source code using the built-in ast module."""

    def __init__(self):
        """Initialize parser."""
        self.source_lines: List[str] = []
        self.file_path: str = ""

    def get_language(self) -> str:
        """Return the language this parser handles."""
        return "python"

    def parse_file(self, file_path: str) -> List[Union[Function, Class]]:
        """
        Parse a Python file and extract functions and classes.

        Args:
            file_path: Path to the Python file

        Returns:
            List of Function and Class objects extracted from the file

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

        self.source_lines = source.split("\n")
        self.file_path = file_path
        extracted = []

        # Extract top-level items (functions and classes)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func = self._extract_function(node, depth=0)
                extracted.append(func)
                # Extract nested functions
                extracted.extend(self._extract_nested_functions(node, func.name, depth=1))
            elif isinstance(node, ast.AsyncFunctionDef):
                func = self._extract_function(node, depth=0, is_async=True)
                extracted.append(func)
                # Extract nested functions
                extracted.extend(self._extract_nested_functions(node, func.name, depth=1))
            elif isinstance(node, ast.ClassDef):
                cls = self._extract_class(node)
                extracted.append(cls)
                # Extract methods from class
                extracted.extend(self._extract_methods(node))

        return extracted

    def _extract_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        depth: int = 0,
        class_name: Optional[str] = None,
        parent_function: Optional[str] = None,
        is_async: bool = False,
    ) -> Function:
        """
        Extract metadata for a single function.

        Args:
            node: AST FunctionDef or AsyncFunctionDef node
            depth: Nesting depth (0 for top-level)
            class_name: Name of parent class (if method)
            parent_function: Name of parent function (if nested)
            is_async: Whether this is an async function

        Returns:
            Function object with extracted metadata
        """
        # Extract source code
        start_line = node.lineno - 1  # ast uses 1-based indexing
        end_line = node.end_lineno if node.end_lineno else len(self.source_lines)
        source_code = "\n".join(self.source_lines[start_line:end_line])

        # Extract signature
        signature = self._get_signature(node, is_async)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract function calls
        calls_to = self._get_calls(node)

        # Determine if this is a method
        is_method = class_name is not None

        return Function(
            name=node.name,
            file_path=self.file_path,
            language="python",
            source_code=source_code,
            docstring=docstring,
            line_number=node.lineno,
            end_line=end_line,
            signature=signature,
            class_name=class_name,
            is_method=is_method,
            is_async=is_async,
            parent_function=parent_function,
            depth=depth,
            calls_to=list(calls_to),
            called_by=[],  # Populated in call graph phase
        )

    def _get_signature(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool = False
    ) -> str:
        """
        Extract function signature.

        Args:
            node: AST FunctionDef or AsyncFunctionDef node
            is_async: Whether this is an async function

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

        prefix = "async def" if is_async else "def"
        return f"{prefix} {node.name}({', '.join(arg_strs_with_hints)}){return_type}:"

    def _get_calls(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Set[str]:
        """
        Extract functions called by this function.

        Args:
            node: AST FunctionDef or AsyncFunctionDef node

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

    def _extract_class(self, node: ast.ClassDef) -> Class:
        """
        Extract metadata for a class and its methods.

        Args:
            node: AST ClassDef node

        Returns:
            Class object with extracted metadata
        """
        # Extract source code
        start_line = node.lineno - 1
        end_line = node.end_lineno if node.end_lineno else len(self.source_lines)
        source_code = "\n".join(self.source_lines[start_line:end_line])

        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base))

        # Extract docstring
        docstring = ast.get_docstring(node)

        return Class(
            name=node.name,
            file_path=self.file_path,
            language="python",
            source_code=source_code,
            docstring=docstring,
            line_number=node.lineno,
            end_line=end_line,
            bases=bases,
        )

    def _extract_nested_functions(
        self,
        parent_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        parent_name: str,
        depth: int = 1,
    ) -> List[Function]:
        """
        Extract nested functions from a function body.

        Args:
            parent_node: Parent function AST node
            parent_name: Name of parent function
            depth: Current nesting depth

        Returns:
            List of nested Function objects
        """
        nested = []

        for node in parent_node.body:
            if isinstance(node, ast.FunctionDef):
                func = self._extract_function(
                    node, depth=depth, parent_function=parent_name
                )
                nested.append(func)
                # Recursively extract nested functions
                nested.extend(
                    self._extract_nested_functions(node, func.name, depth=depth + 1)
                )
            elif isinstance(node, ast.AsyncFunctionDef):
                func = self._extract_function(
                    node, depth=depth, parent_function=parent_name, is_async=True
                )
                nested.append(func)
                # Recursively extract nested functions
                nested.extend(
                    self._extract_nested_functions(node, func.name, depth=depth + 1)
                )

        return nested

    def _extract_methods(self, class_node: ast.ClassDef) -> List[Function]:
        """
        Extract methods from a class.

        Args:
            class_node: AST ClassDef node

        Returns:
            List of Function objects representing methods
        """
        methods = []

        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                func = self._extract_function(node, class_name=class_node.name)
                methods.append(func)
                # Extract nested functions within methods
                methods.extend(
                    self._extract_nested_functions(node, func.name, depth=2)
                )
            elif isinstance(node, ast.AsyncFunctionDef):
                func = self._extract_function(
                    node, class_name=class_node.name, is_async=True
                )
                methods.append(func)
                # Extract nested functions within methods
                methods.extend(
                    self._extract_nested_functions(node, func.name, depth=2)
                )

        return methods
