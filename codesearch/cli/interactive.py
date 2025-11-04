"""Interactive CLI features for enhanced code exploration."""

import typer
import lancedb
from pathlib import Path
from typing import Optional, List
import logging

from codesearch.query import QueryEngine
from codesearch.query.models import SearchResult
from codesearch.cli.config import get_db_path, validate_db_exists, get_config
from codesearch.cli.formatting import (
    format_results_table,
    display_code_snippet,
    display_call_graph,
    highlight_code,
)

logger = logging.getLogger(__name__)


class Paginator:
    """Utility for paginating results."""

    def __init__(self, items: List, page_size: int = 10):
        """Initialize paginator.

        Args:
            items: List of items to paginate
            page_size: Number of items per page
        """
        self.items = items
        self.page_size = page_size
        self.total_pages = (len(items) + page_size - 1) // page_size
        self.current_page = 1

    def get_page(self, page_num: int) -> List:
        """Get items for a specific page.

        Args:
            page_num: Page number (1-indexed)

        Returns:
            List of items for the page
        """
        if page_num < 1 or page_num > self.total_pages:
            return []
        start = (page_num - 1) * self.page_size
        end = start + self.page_size
        return self.items[start:end]

    def next_page(self) -> List:
        """Get next page of items."""
        if self.current_page < self.total_pages:
            self.current_page += 1
        return self.get_page(self.current_page)

    def prev_page(self) -> List:
        """Get previous page of items."""
        if self.current_page > 1:
            self.current_page -= 1
        return self.get_page(self.current_page)


def detail(
    entity_name: str = typer.Argument(..., help="Name of entity to show details for"),
) -> None:
    """Drill down into function/class details.

    Shows detailed information about a specific code entity including:
    - Full source code with syntax highlighting
    - Type and language information
    - File location and line numbers
    - Associated relationships

    Example:
        $ codesearch detail parse_request
        $ codesearch detail MyClass
    """
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)

        try:
            # Query for the entity
            entities_table = client.open_table("code_entities")
            all_entities = entities_table.search().to_list()

            # Find matching entity
            matching = [e for e in all_entities if e.get("name") == entity_name]

            if not matching:
                typer.echo(f"❌ Entity '{entity_name}' not found in database")
                raise typer.Exit(2)

            entity = matching[0]

            # Display entity details
            typer.echo(f"\n📋 Entity Details: {entity.get('name', 'unknown')}")
            typer.echo(f"{'─' * 60}")

            typer.echo(f"Type: {entity.get('entity_type', 'unknown')}")
            typer.echo(f"Language: {entity.get('language', 'unknown')}")
            typer.echo(f"File: {entity.get('file_path', 'unknown')}")
            typer.echo(
                f"Lines: {entity.get('start_line', '?')}-{entity.get('end_line', '?')}"
            )

            # Display source code if available
            code = entity.get("code_text")
            if code:
                typer.echo(f"\n{'─' * 60}")
                typer.echo("Source Code:")
                typer.echo("─" * 60)
                language = entity.get("language", "python")
                start_line = entity.get("start_line", 1)
                highlighted = display_code_snippet(code, language, start_line)
                typer.echo(highlighted)

            typer.echo(f"{'─' * 60}\n")

        except Exception as e:
            typer.echo(f"⚠️  Error retrieving entity details: {e}", err=True)
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def context(
    entity_name: str = typer.Argument(..., help="Name of entity to analyze"),
) -> None:
    """Show function context (callers and callees).

    Displays the call graph for a specific entity, showing:
    - Functions this entity calls
    - Functions that call this entity
    - Relationship types

    Example:
        $ codesearch context main
        $ codesearch context process_data
    """
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)

        try:
            # Get relationships
            relationships_table = client.open_table("code_relationships")
            all_rels = relationships_table.search().to_list()

            # Filter for this entity
            outgoing = [r for r in all_rels if r.get("source_id") == entity_name]
            incoming = [r for r in all_rels if r.get("target_id") == entity_name]

            if not outgoing and not incoming:
                typer.echo(f"⚠️  No relationships found for '{entity_name}'")
                raise typer.Exit(0)

            typer.echo(f"\n📊 Context for '{entity_name}'")
            typer.echo(f"{'─' * 60}")

            if outgoing:
                typer.echo(f"\n🔹 This entity calls ({len(outgoing)}):")
                for rel in outgoing[:20]:
                    target = rel.get("target_id", "unknown")
                    rel_type = rel.get("relationship_type", "calls")
                    typer.echo(f"  ➜ {target} ({rel_type})")

            if incoming:
                typer.echo(f"\n🔹 Called by ({len(incoming)}):")
                for rel in incoming[:20]:
                    source = rel.get("source_id", "unknown")
                    rel_type = rel.get("relationship_type", "calls")
                    typer.echo(f"  ← {source} ({rel_type})")

            if len(outgoing) > 20 or len(incoming) > 20:
                typer.echo("\n... (showing first 20 of each)")

            typer.echo(f"{'─' * 60}\n")

        except Exception as e:
            typer.echo(f"⚠️  Error retrieving relationships: {e}", err=True)
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def compare(
    entity1: str = typer.Argument(..., help="First entity to compare"),
    entity2: str = typer.Argument(..., help="Second entity to compare"),
) -> None:
    """Compare two functions or classes.

    Shows detailed comparison between two entities:
    - Similarity score
    - Size (lines of code)
    - Language and type
    - Relationship overview

    Example:
        $ codesearch compare parse_json parse_xml
        $ codesearch compare validate_email check_email
    """
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)

        try:
            entities_table = client.open_table("code_entities")
            all_entities = entities_table.search().to_list()

            # Find both entities
            ent1 = next((e for e in all_entities if e.get("name") == entity1), None)
            ent2 = next((e for e in all_entities if e.get("name") == entity2), None)

            if not ent1:
                typer.echo(f"❌ Entity '{entity1}' not found", err=True)
                raise typer.Exit(2)
            if not ent2:
                typer.echo(f"❌ Entity '{entity2}' not found", err=True)
                raise typer.Exit(2)

            # Display comparison
            typer.echo(f"\n⚖️  Comparison: {entity1} vs {entity2}")
            typer.echo(f"{'─' * 60}")

            # Header
            typer.echo(
                f"{'Property':<20} {'':20} | {'Entity 1':<20} | {'Entity 2':<20}"
            )
            typer.echo(f"{'─' * 60}")

            # Type
            type1 = ent1.get("entity_type", "unknown")
            type2 = ent2.get("entity_type", "unknown")
            match = "✓" if type1 == type2 else "✗"
            typer.echo(f"Type {match:<18} | {type1:<20} | {type2:<20}")

            # Language
            lang1 = ent1.get("language", "unknown")
            lang2 = ent2.get("language", "unknown")
            match = "✓" if lang1 == lang2 else "✗"
            typer.echo(f"Language {match:<14} | {lang1:<20} | {lang2:<20}")

            # Lines of code
            lines1 = (ent1.get("end_line", 0) or 0) - (ent1.get("start_line", 0) or 0)
            lines2 = (ent2.get("end_line", 0) or 0) - (ent2.get("start_line", 0) or 0)
            typer.echo(f"Lines of code       | {lines1:<20} | {lines2:<20}")

            # File
            file1 = ent1.get("file_path", "unknown")
            file2 = ent2.get("file_path", "unknown")
            match = "✓" if file1 == file2 else "✗"
            typer.echo(f"Same file {match:<14} | {file1:<20} | {file2:<20}")

            typer.echo(f"{'─' * 60}\n")

        except Exception as e:
            typer.echo(f"⚠️  Error comparing entities: {e}", err=True)
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def config_show() -> None:
    """Show current configuration settings.

    Displays all active configuration settings from:
    - Environment variables
    - Configuration files
    - Default values

    Example:
        $ codesearch config-show
    """
    try:
        config = get_config()

        typer.echo("\n⚙️  Current Configuration")
        typer.echo(f"{'─' * 60}")

        for key, value in sorted(config.items()):
            typer.echo(f"{key:<20} : {value}")

        typer.echo(f"{'─' * 60}\n")

    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def config_init(
    config_path: Optional[str] = typer.Option(
        None, "--path", "-p", help="Custom config file path"
    ),
) -> None:
    """Initialize a new configuration file.

    Creates a default configuration file in the specified location.
    If no path is provided, uses default location (~/.codesearch/config.json).

    Example:
        $ codesearch config-init
        $ codesearch config-init --path ./codesearch.json
    """
    try:
        from codesearch.cli.config import init_config, DEFAULT_CONFIG

        if config_path:
            path = Path(config_path)
        else:
            path = Path.home() / ".codesearch" / "config.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            if not typer.confirm(f"File {path} already exists. Overwrite?"):
                typer.echo("❌ Cancelled")
                raise typer.Exit(1)

        result = init_config(path)

        if result:
            typer.echo(f"✅ Configuration initialized at {path}")
            typer.echo(f"📝 Default settings:")
            for key, value in DEFAULT_CONFIG.items():
                typer.echo(f"   {key}: {value}")
        else:
            typer.echo(f"❌ Failed to initialize configuration")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)
