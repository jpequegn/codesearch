"""CLI commands for codesearch."""

import typer
import lancedb
from typing import Optional
from codesearch.query import QueryEngine
from codesearch.cli.config import get_db_path, validate_db_exists
from codesearch.cli.formatting import format_results_json, format_results_table


def pattern(
    query: str = typer.Argument(..., help="Semantic search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to return"),
    output: str = typer.Option("table", "--output", "-o", help="Output format (table or json)"),
) -> None:
    """Search for code matching a semantic description."""
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            typer.echo("Run 'codesearch index <path>' to create a database", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)
        engine = QueryEngine(client)
        results = engine.search_text(query, limit=limit)

        if not results:
            typer.echo("No results found")
            raise typer.Exit(0)

        if output == "json":
            typer.echo(format_results_json(results))
        else:
            typer.echo(format_results_table(results))

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def find_similar(
    entity_name: str = typer.Argument(..., help="Entity name to find similar matches for"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to return"),
    language: Optional[str] = typer.Option(None, "--language", "-L", help="Filter by language"),
) -> None:
    """Find functions similar to the given entity."""
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)
        engine = QueryEngine(client)

        # TODO: Get entity by name and extract its vector
        # For now, use a dummy vector
        query_vector = [0.1] * 768

        results = engine.search_vector(query_vector, limit=limit + 1)
        # Filter out the original entity
        results = [r for r in results if r.name != entity_name][:limit]

        if not results:
            typer.echo("No similar results found")
            raise typer.Exit(0)

        typer.echo(format_results_table(results))

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def dependencies(
    entity_name: str = typer.Argument(..., help="Entity to analyze"),
    direction: str = typer.Option(
        "both", "--direction", "-d", help="Direction: 'calls', 'callers', or 'both'"
    ),
) -> None:
    """Show call graph for an entity."""
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)

        # TODO: Implement call graph querying from code_relationships table
        typer.echo(f"📊 Dependencies for '{entity_name}' ({direction})")
        typer.echo("(Feature coming soon)")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def index(
    path: str = typer.Argument(..., help="Path to codebase or file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-index"),
    language: Optional[str] = typer.Option(None, "--language", "-L", help="Language filter"),
) -> None:
    """Index a codebase."""
    try:
        db_path = get_db_path()

        # Create database directory if needed
        import os

        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        client = lancedb.connect(db_path)

        # TODO: Implement DataIngestionPipeline integration
        typer.echo(f"📇 Indexing {path}...")
        typer.echo("(Feature coming soon)")

    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def refactor_dupes(
    threshold: float = typer.Option(
        0.95, "--threshold", "-t", help="Similarity threshold (0.0-1.0)"
    )
) -> None:
    """Find duplicate or very similar code."""
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)

        # TODO: Implement pairwise similarity search for duplicates
        typer.echo(f"🔍 Finding duplicates with threshold {threshold}...")
        typer.echo("(Feature coming soon)")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)
