"""CLI entry point for codesearch.

This module provides the main CLI interface using Typer framework.
Supports semantic code search, analysis, and indexing operations.
"""

import typer
from typing import Annotated, Optional
from codesearch import __version__
from codesearch.cli.commands import (
    pattern, find_similar, dependencies, index, refactor_dupes, list_functions,
    repo_list, repo_add, repo_remove, search_multi, benchmark_models
)
from codesearch.cli.interactive import (
    detail, context, compare, config_show, config_init
)


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


# Create Typer application with enhanced configuration
app = typer.Typer(
    help="Semantic code search tool using vector embeddings and LanceDB",
    no_args_is_help=True,
)


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-v", callback=version_callback, is_eager=True,
                     help="Show version and exit")
    ] = None,
) -> None:
    """Semantic code search using vector embeddings and LanceDB."""
    pass

# Register all commands as subcommands
app.command(name="search")(pattern)
app.command(name="find-similar")(find_similar)
app.command(name="deps")(dependencies)
app.command(name="index")(index)
app.command(name="refactor-dupes")(refactor_dupes)
app.command(name="list-functions")(list_functions)

# Register interactive commands
app.command(name="detail")(detail)
app.command(name="context")(context)
app.command(name="compare")(compare)
app.command(name="config-show")(config_show)
app.command(name="config-init")(config_init)

# Register repository management commands
app.command(name="repo-list")(repo_list)
app.command(name="repo-add")(repo_add)
app.command(name="repo-remove")(repo_remove)
app.command(name="search-multi")(search_multi)

# Register benchmark commands
app.command(name="benchmark-models")(benchmark_models)


def run() -> None:
    """Entry point function for CLI."""
    app()


if __name__ == "__main__":
    run()
