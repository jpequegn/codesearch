"""CLI entry point for codesearch."""

import typer
from codesearch.cli.commands import pattern, find_similar, dependencies, index, refactor_dupes

app = typer.Typer(help="Semantic code search tool")

# Register all commands
app.command()(pattern)
app.command(name="find-similar")(find_similar)
app.command()(dependencies)
app.command()(index)
app.command(name="refactor-dupes")(refactor_dupes)

if __name__ == "__main__":
    app()
