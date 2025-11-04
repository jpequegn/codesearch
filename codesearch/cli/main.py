"""CLI entry point for codesearch.

This module provides the main CLI interface using Typer framework.
Supports semantic code search, analysis, and indexing operations.
"""

import typer
from typing import Optional
from codesearch import __version__
from codesearch.cli.commands import pattern, find_similar, dependencies, index, refactor_dupes


def version_callback(value: bool) -> None:
    """Display version information and exit.

    Args:
        value: Whether version flag was set

    Raises:
        typer.Exit: Always exits after displaying version
    """
    if value:
        typer.echo(f"codesearch version {__version__}")
        raise typer.Exit()


def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Codesearch - Semantic code search using vector embeddings.

    A powerful CLI tool for semantic code search using vector embeddings
    and LanceDB. Search for similar code, analyze dependencies, and
    optimize duplicate code patterns.
    """
    # This function serves as the main entry point for the CLI app
    pass


# Create Typer application with enhanced configuration
app = typer.Typer(
    help="Semantic code search tool using vector embeddings and LanceDB",
    invoke_without_command=True,
    no_args_is_help=True,
)

# Add version callback to main app
app.command()(main)

# Register all commands as subcommands
app.command(name="search")(pattern)
app.command(name="find-similar")(find_similar)
app.command(name="deps")(dependencies)
app.command(name="index")(index)
app.command(name="refactor-dupes")(refactor_dupes)


def run() -> None:
    """Entry point function for CLI."""
    app()


if __name__ == "__main__":
    run()
