"""CLI entry point for codesearch."""

import typer
from codesearch.cli.commands import (
    pattern, find_similar, dependencies, index, refactor_dupes, list_functions
)

app = typer.Typer(help="Semantic code search tool")

# Register all commands
app.command(name="search")(pattern)
app.command(name="find-similar")(find_similar)
app.command(name="deps")(dependencies)
app.command()(index)
app.command(name="refactor-dupes")(refactor_dupes)
app.command(name="list-functions")(list_functions)

if __name__ == "__main__":
    app()
