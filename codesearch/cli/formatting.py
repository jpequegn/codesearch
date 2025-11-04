"""Output formatting utilities for CLI with Rich integration."""

import json
from typing import List, Optional
from io import StringIO
from rich.table import Table
from rich.console import Console
from rich.syntax import Syntax
from rich.tree import Tree
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from codesearch.query.models import SearchResult


def format_results_json(results: List[SearchResult]) -> str:
    """Format results as JSON.

    Args:
        results: List of search results

    Returns:
        JSON string representation
    """
    data = [
        {
            "entity_id": r.entity_id,
            "name": r.name,
            "language": r.language,
            "file_path": r.file_path,
            "similarity_score": r.similarity_score,
            "entity_type": r.entity_type,
            "repository": r.repository,
            "start_line": r.start_line,
            "end_line": r.end_line,
        }
        for r in results
    ]
    return json.dumps(data, indent=2)


def format_results_table(results: List[SearchResult]) -> str:
    """Format results as a rich ASCII table.

    Args:
        results: List of search results

    Returns:
        Formatted table string
    """
    if not results:
        return "No results found"

    # Create a rich table
    table = Table(title="Search Results", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan", width=20)
    table.add_column("Type", style="green", width=12)
    table.add_column("Language", style="yellow", width=12)
    table.add_column("File", style="blue", width=25)
    table.add_column("Score", style="red", width=8)

    for r in results:
        score_str = f"{r.similarity_score:.2%}"
        table.add_row(
            r.name[:20],
            r.entity_type[:12],
            r.language[:12],
            r.file_path[:25],
            score_str,
        )

    # Render to string using StringIO
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=120)
    console.print(table)
    return buffer.getvalue().rstrip()


def format_results(results: List[SearchResult], format: str = "table") -> str:
    """Format results in the specified format.

    Args:
        results: List of search results
        format: Output format ('table' or 'json')

    Returns:
        Formatted results string
    """
    if format == "json":
        return format_results_json(results)
    else:
        return format_results_table(results)


def highlight_code(code: str, language: str = "python") -> str:
    """Highlight code with syntax coloring.

    Args:
        code: Source code to highlight
        language: Programming language for syntax highlighting

    Returns:
        Syntax-highlighted code string
    """
    try:
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True, width=120)
        console.print(syntax)
        return buffer.getvalue().rstrip()
    except Exception:
        # Fallback to plain code if highlighting fails
        return code


def display_code_snippet(code: str, language: str = "python", start_line: int = 1) -> str:
    """Display a code snippet with highlighting and context.

    Args:
        code: Source code
        language: Programming language
        start_line: Starting line number for context

    Returns:
        Formatted code snippet string
    """
    try:
        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=True,
            start_line=start_line,
            highlight_lines={start_line},
        )
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True, width=120)
        console.print(syntax)
        return buffer.getvalue().rstrip()
    except Exception:
        # Fallback to plain code
        lines = code.split("\n")
        numbered_lines = [f"{start_line + i:4d} | {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)


def display_call_graph(relationships: List[dict]) -> str:
    """Display call graph relationships as an ASCII tree.

    Args:
        relationships: List of caller/callee relationships

    Returns:
        Formatted call graph string
    """
    if not relationships:
        return "No relationships found"

    # Build a tree structure from relationships
    # Group by caller to show a hierarchical structure
    tree = Tree("📊 Call Graph")

    # Group relationships by source
    grouped = {}
    for rel in relationships:
        source = rel.get("source_id") or rel.get("caller", "unknown")
        target = rel.get("target_id") or rel.get("callee", "unknown")
        rel_type = rel.get("relationship_type", "calls")

        if source not in grouped:
            grouped[source] = []
        grouped[source].append((target, rel_type))

    # Add nodes to tree
    for source, targets in grouped.items():
        source_node = tree.add(f"[cyan]{source}[/cyan]")
        for target, rel_type in targets:
            source_node.add(f"[green]→ {target}[/green] ({rel_type})")

    # Render to string using StringIO
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=120)
    console.print(tree)
    return buffer.getvalue().rstrip()


class ProgressTracker:
    """Context manager for tracking progress during long operations."""

    def __init__(self, description: str = "Processing", total: int = 100):
        """Initialize progress tracker.

        Args:
            description: Description of the operation
            total: Total number of items to process
        """
        self.description = description
        self.total = total
        self.progress = None

    def __enter__(self):
        """Start progress tracking."""
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TimeRemainingColumn(),
        )
        self.progress.__enter__()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop progress tracking."""
        if self.progress:
            self.progress.__exit__(exc_type, exc_val, exc_tb)

    def update(self, advance: int = 1, description: Optional[str] = None):
        """Update progress.

        Args:
            advance: Number of items to advance
            description: Optional new description
        """
        if self.progress:
            self.progress.update(self.task_id, advance=advance, description=description or self.description)
