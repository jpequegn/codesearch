"""Data models for codesearch."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Class:
    """Represents a class extracted from source code."""

    name: str
    file_path: str
    language: str = "python"

    # Code content
    source_code: str = ""
    docstring: Optional[str] = None

    # Metadata
    line_number: int = 0
    end_line: int = 0

    # Base classes
    bases: List[str] = field(default_factory=list)

    # Embedding
    embedding: Optional[List[float]] = None

    def __hash__(self):
        """Make Class hashable for use in sets."""
        return hash((self.name, self.file_path, self.line_number))

    def __eq__(self, other):
        """Compare classes by identity."""
        if not isinstance(other, Class):
            return False
        return (
            self.name == other.name
            and self.file_path == other.file_path
            and self.line_number == other.line_number
        )


@dataclass
class Function:
    """Represents a function extracted from source code."""

    name: str
    file_path: str
    language: str = "python"

    # Code content
    source_code: str = ""
    docstring: Optional[str] = None

    # Metadata
    line_number: int = 0
    end_line: int = 0
    signature: str = ""

    # Class context (if method)
    class_name: Optional[str] = None
    is_method: bool = False
    is_async: bool = False

    # Nesting information
    parent_function: Optional[str] = None  # For nested functions
    depth: int = 0  # 0 for top-level, 1+ for nested

    # Call graph
    calls_to: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)

    # Embedding
    embedding: Optional[List[float]] = None

    def __hash__(self):
        """Make Function hashable for use in sets."""
        return hash((self.name, self.file_path, self.line_number))

    def __eq__(self, other):
        """Compare functions by identity."""
        if not isinstance(other, Function):
            return False
        return (
            self.name == other.name
            and self.file_path == other.file_path
            and self.line_number == other.line_number
        )

    def fully_qualified_name(self) -> str:
        """Return fully qualified name for nested functions/methods."""
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        elif self.parent_function:
            return f"{self.parent_function}.{self.name}"
        return self.name


@dataclass
class Import:
    """Represents an import statement in source code."""

    module: str  # e.g., 'os', 'mymodule.submodule'
    names: Dict[str, Optional[str]]  # Maps imported name to alias (None if no alias)
    # e.g., {'func': None} for 'from x import func'
    # e.g., {'func': 'renamed'} for 'from x import func as renamed'
    # e.g., {'os': None} for 'import os'
    import_type: str  # 'import' or 'from'
    file_path: str
    line_number: int

    def get_original_name(self, alias: str) -> Optional[str]:
        """Get the original name that was imported with a given alias.

        Args:
            alias: The imported name (possibly aliased)

        Returns:
            The original module.name or None if not found
        """
        for original, actual_alias in self.names.items():
            if actual_alias == alias or (actual_alias is None and original == alias):
                if self.import_type == "from":
                    return f"{self.module}.{original}"
                else:
                    return original
        return None


@dataclass
class CallGraph:
    """Represents the call graph for a codebase."""

    functions: Dict[str, Function] = field(default_factory=dict)  # keyed by fully_qualified_name
    classes: Dict[str, Class] = field(default_factory=dict)
    imports: Dict[str, List[Import]] = field(default_factory=dict)  # keyed by file_path

    def add_function(self, func: Function) -> None:
        """Add a function to the graph."""
        key = f"{func.file_path}:{func.line_number}"
        self.functions[key] = func

    def add_class(self, cls: Class) -> None:
        """Add a class to the graph."""
        key = f"{cls.file_path}:{cls.line_number}"
        self.classes[key] = cls

    def add_imports(self, file_path: str, imports: List[Import]) -> None:
        """Add imports for a file."""
        self.imports[file_path] = imports

    def resolve_call(
        self, caller_file: str, call_name: str, imports: List[Import]
    ) -> Optional[Tuple[str, Function]]:
        """Resolve a function call to an actual Function object.

        Args:
            caller_file: Path of the file containing the call
            call_name: Name of the function being called
            imports: List of imports in the caller file

        Returns:
            Tuple of (function_key, Function) or None if not found
        """
        # Try to find in same file first
        for func_key, func in self.functions.items():
            if func.file_path == caller_file and func.name == call_name:
                return (func_key, func)

        # Try to resolve through imports
        for imp in imports:
            if call_name in imp.names:
                # Check if this import provides this function
                if imp.import_type == "from":
                    full_name = f"{imp.module}.{call_name}"
                    # Try to find function from that module
                    for func_key, func in self.functions.items():
                        if func.name == call_name and func.file_path.endswith(
                            imp.module.replace(".", "/") + ".py"
                        ):
                            return (func_key, func)

        return None

    def build_called_by(self) -> None:
        """Build the called_by relationships for all functions."""
        # Clear existing called_by
        for func in self.functions.values():
            func.called_by = []

        # For each function, resolve its calls
        for caller_key, caller in self.functions.items():
            caller_file = caller.file_path
            caller_imports = self.imports.get(caller_file, [])

            for call_name in caller.calls_to:
                result = self.resolve_call(caller_file, call_name, caller_imports)
                if result:
                    callee_key, callee = result
                    # Add caller to callee's called_by list
                    if caller_key not in callee.called_by:
                        callee.called_by.append(caller_key)

    def get_call_chain(self, start_func_key: str) -> Set[str]:
        """Get all functions transitively called by a given function.

        Args:
            start_func_key: Function to start from

        Returns:
            Set of function keys that are called (directly or indirectly)
        """
        visited = set()
        stack = [start_func_key]

        while stack:
            func_key = stack.pop()
            if func_key in visited:
                continue
            visited.add(func_key)

            func = self.functions.get(func_key)
            if func:
                # Resolve all calls made by this function
                caller_file = func.file_path
                caller_imports = self.imports.get(caller_file, [])

                for call_name in func.calls_to:
                    result = self.resolve_call(caller_file, call_name, caller_imports)
                    if result:
                        callee_key, _ = result
                        if callee_key not in visited:
                            stack.append(callee_key)

        return visited - {start_func_key}  # Exclude the starting function

    def get_callers(self, func_key: str) -> Set[str]:
        """Get all functions that call a given function (directly).

        Args:
            func_key: Function to get callers for

        Returns:
            Set of function keys that call this function
        """
        func = self.functions.get(func_key)
        if func:
            return set(func.called_by)
        return set()


@dataclass
class FileMetadata:
    """Metadata about a source code file."""

    file_path: str
    language: str
    size_bytes: int
    modified_time: datetime
    is_ignored: bool = False
    ignore_reason: Optional[str] = None

    def relative_path(self, base_path: str) -> str:
        """Get relative path from base directory.

        Args:
            base_path: Base directory path

        Returns:
            Relative path from base
        """
        if self.file_path.startswith(base_path):
            return self.file_path[len(base_path):].lstrip("/")
        return self.file_path


@dataclass
class RepositoryScanner:
    """Configuration and state for scanning repositories."""

    repositories: Dict[str, str] = field(default_factory=dict)  # name -> path
    supported_languages: Set[str] = field(default_factory=lambda: {"python"})
    ignored_directories: Set[str] = field(
        default_factory=lambda: {
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            "node_modules",
            ".egg-info",
            "dist",
            "build",
            ".pytest_cache",
            ".mypy_cache",
            ".tox",
        }
    )
    ignored_file_patterns: Set[str] = field(
        default_factory=lambda: {
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".DS_Store",
            "*.egg-info",
            ".coverage",
        }
    )

    def add_repository(self, name: str, path: str) -> None:
        """Add a repository to scan.

        Args:
            name: Name/identifier for the repository
            path: File system path to the repository
        """
        self.repositories[name] = path

    def should_scan_directory(self, dir_name: str) -> bool:
        """Check if a directory should be scanned.

        Args:
            dir_name: Directory name (not full path)

        Returns:
            True if directory should be scanned
        """
        return dir_name not in self.ignored_directories

    def should_scan_file(self, file_path: str, language: str) -> bool:
        """Check if a file should be scanned.

        Args:
            file_path: Full file path
            language: Detected language for file

        Returns:
            True if file should be scanned
        """
        # Check if language is supported
        if language not in self.supported_languages:
            return False

        # Check if file matches ignored patterns
        file_name = file_path.split("/")[-1]
        for pattern in self.ignored_file_patterns:
            if pattern.startswith("*."):
                ext = pattern[2:]
                if file_name.endswith(f".{ext}"):
                    return False
            elif file_name == pattern:
                return False

        return True

    def get_language_from_extension(self, file_path: str) -> Optional[str]:
        """Detect language from file extension.

        Args:
            file_path: File path

        Returns:
            Language name or None if unknown
        """
        extension_map = {
            ".py": "python",
            ".pyx": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
        }

        for ext, lang in extension_map.items():
            if file_path.endswith(ext):
                return lang

        return None


@dataclass
class EmbeddingModel:
    """Configuration for an embedding model."""

    name: str                      # e.g., "codebert-base"
    model_name: str                # HuggingFace model ID (e.g., "microsoft/codebert-base")
    dimensions: int                # Vector size (e.g., 768)
    max_length: int                # Max input tokens (e.g., 512)
    device: str = "auto"           # "cpu", "cuda", or "auto"
