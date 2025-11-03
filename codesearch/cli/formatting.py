"""Output formatting utilities for CLI."""

import json
from typing import List
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
    """Format results as ASCII table.

    Args:
        results: List of search results

    Returns:
        Formatted table string
    """
    if not results:
        return "No results found"

    lines = []
    lines.append("┌─────────────┬──────────┬──────────┬─────────┐")
    lines.append("│ Name        │ Language │ File     │ Score   │")
    lines.append("├─────────────┼──────────┼──────────┼─────────┤")

    for r in results:
        name = r.name[:11].ljust(11)
        lang = r.language[:8].ljust(8)
        file = r.file_path[:8].ljust(8)
        score = f"{r.similarity_score:.2f}".ljust(7)
        lines.append(f"│ {name} │ {lang} │ {file} │ {score} │")

    lines.append("└─────────────┴──────────┴──────────┴─────────┘")
    return "\n".join(lines)


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


def display_call_graph(relationships: List[dict]) -> str:
    """Display call graph relationships in tree format.

    Args:
        relationships: List of caller/callee relationships

    Returns:
        Formatted call graph string
    """
    if not relationships:
        return "No relationships found"

    lines = []
    lines.append("Call Graph:")
    for rel in relationships:
        lines.append(f"  {rel.get('caller', 'unknown')} → {rel.get('callee', 'unknown')}")
    return "\n".join(lines)
