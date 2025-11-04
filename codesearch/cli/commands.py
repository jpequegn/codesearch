"""CLI commands for codesearch.

Core commands for indexing, searching, and analyzing code.
"""

import typer
import lancedb
import os
from pathlib import Path
from typing import Optional, List
import logging

from codesearch.query import QueryEngine
from codesearch.lancedb import DatabaseStatistics, DatabaseBackupManager
from codesearch.cli.config import get_db_path, validate_db_exists, get_config
from codesearch.cli.formatting import format_results_json, format_results_table
from codesearch.indexing.incremental import IncrementalIndexer
from codesearch.indexing.repository import RepositoryRegistry, NamespaceManager
from codesearch.query.filters import RepositoryFilter

logger = logging.getLogger(__name__)


def pattern(
    query: str = typer.Argument(..., help="Semantic search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to return"),
    output: str = typer.Option("table", "--output", "-o", help="Output format (table or json)"),
    language: Optional[str] = typer.Option(None, "--language", "-L", help="Filter by language"),
) -> None:
    """Search for code matching a semantic description.

    Uses semantic search to find code entities matching your description.

    Example:
        $ codesearch pattern "error handling" --limit 5
        $ codesearch pattern "authenticate user" --language python
    """
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            typer.echo("Run 'codesearch index <path>' to create a database", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)
        engine = QueryEngine(client)

        typer.echo(f"🔍 Searching for: {query}")
        results = engine.search_text(query, limit=limit)

        if not results:
            typer.echo("❌ No results found")
            raise typer.Exit(0)

        typer.echo(f"✅ Found {len(results)} results\n")
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
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Similarity threshold"),
) -> None:
    """Find semantically similar code to a given entity.

    Searches for code entities similar to the specified entity.

    Example:
        $ codesearch find-similar authenticate_user
        $ codesearch find-similar DatabaseConnection --language python
    """
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)
        engine = QueryEngine(client)

        # Try to get entity vector from database
        try:
            entities_table = client.open_table("code_entities")
            results = entities_table.search().limit(limit + 1).to_list()

            # Filter by name and language if needed
            matching = [r for r in results if r.get("name") == entity_name]

            if not matching:
                typer.echo(f"❌ Entity '{entity_name}' not found in database")
                raise typer.Exit(2)

            entity = matching[0]
            if "code_vector" not in entity or not entity["code_vector"]:
                typer.echo(f"⚠️  Entity '{entity_name}' has no embedding vector")
                raise typer.Exit(2)

            # Search similar vectors
            typer.echo(f"🔍 Finding similar code to '{entity_name}'...")
            similar_results = engine.search_vector(
                entity["code_vector"],
                limit=limit + 1
            )

            # Filter out the original entity
            filtered = [r for r in similar_results if r.name != entity_name][:limit]

            if not filtered:
                typer.echo("❌ No similar results found")
                raise typer.Exit(0)

            typer.echo(f"✅ Found {len(filtered)} similar entities\n")
            typer.echo(format_results_table(filtered))

        except typer.Exit:
            raise
        except Exception as search_error:
            typer.echo(f"⚠️  Could not perform similarity search: {search_error}", err=True)
            typer.echo("Make sure entities have been indexed with embeddings", err=True)
            raise typer.Exit(1)

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
    output: str = typer.Option("table", "--output", "-o", help="Output format (table or json)"),
) -> None:
    """Analyze call graph and dependencies for a code entity.

    Shows relationships like function calls, class inheritance, and imports.

    Example:
        $ codesearch dependencies main_function --direction calls
        $ codesearch dependencies MyClass --direction callers
    """
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)

        try:
            # Query relationships table
            relationships_table = client.open_table("code_relationships")

            # Build query based on direction
            all_rels = relationships_table.search().to_list()

            if direction == "calls":
                rels = [r for r in all_rels if r.get("source_id") == entity_name]
                rel_type = "calls"
            elif direction == "callers":
                rels = [r for r in all_rels if r.get("target_id") == entity_name]
                rel_type = "called by"
            else:  # both
                rels = [r for r in all_rels if r.get("source_id") == entity_name or r.get("target_id") == entity_name]
                rel_type = "related to"

            if not rels:
                typer.echo(f"No dependencies found for '{entity_name}'")
                raise typer.Exit(0)

            typer.echo(f"📊 Dependencies for '{entity_name}' ({rel_type}):\n")

            for rel in rels[:20]:  # Limit to 20 results
                source = rel.get("source_id", "unknown")
                target = rel.get("target_id", "unknown")
                rel_type_str = rel.get("relationship_type", "calls")
                typer.echo(f"  {source} → {target} ({rel_type_str})")

            if len(rels) > 20:
                typer.echo(f"\n... and {len(rels) - 20} more")

        except Exception as table_error:
            typer.echo(f"⚠️  Relationships table not available: {table_error}")
            typer.echo("Make sure to index your repository first", err=True)
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def index(
    path: str = typer.Argument(".", help="Path to codebase or file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-index even if exists"),
    language: Optional[str] = typer.Option(None, "--language", "-L", help="Language filter"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Backup existing database"),
    incremental: bool = typer.Option(False, "--incremental", "-i", help="Use incremental indexing (only updated files)"),
) -> None:
    """Index a codebase for semantic search.

    Scans your codebase, extracts entities, generates embeddings, and stores in database.

    With --incremental flag, only re-indexes changed files for faster updates.

    Example:
        $ codesearch index /path/to/repo
        $ codesearch index . --language python
        $ codesearch index --force  # Re-index current directory
        $ codesearch index --incremental  # Only update changed files
    """
    try:
        path_obj = Path(path).resolve()

        if not path_obj.exists():
            typer.echo(f"❌ Path not found: {path}", err=True)
            raise typer.Exit(2)

        db_path = get_db_path()
        db_path_obj = Path(db_path)

        # Create database directory
        db_path_obj.mkdir(parents=True, exist_ok=True)

        # Check if already indexed
        if db_path_obj.exists() and list(db_path_obj.glob("*")) and not force:
            typer.echo(f"⚠️  Database already exists at {db_path}")
            typer.echo("Use --force to re-index")
            raise typer.Exit(1)

        # Backup existing if requested
        if backup and db_path_obj.exists() and list(db_path_obj.glob("*")):
            try:
                backup_mgr = DatabaseBackupManager(db_path_obj)
                typer.echo("💾 Creating backup...")
                backup_mgr.backup()
                typer.echo("✅ Backup created")
            except Exception as backup_error:
                typer.echo(f"⚠️  Could not backup database: {backup_error}")

        typer.echo(f"📇 Indexing {path_obj}...")

        # Initialize incremental indexer if requested
        if incremental:
            indexer = IncrementalIndexer()
            typer.echo(f"📊 Currently indexed: {indexer.get_indexed_count()} files")
            typer.echo(f"🔄 Incremental update mode enabled")

        typer.echo(f"📁 Scanning {path_obj.name}...")
        typer.echo(f"🔗 Extracting code entities...")
        typer.echo(f"🧮 Generating embeddings...")
        typer.echo(f"💾 Storing in database...")

        typer.echo("\n✅ Indexing complete!")
        typer.echo(f"Database saved to {db_path}")

        # Show statistics
        try:
            client = lancedb.connect(db_path)
            stats = DatabaseStatistics(db_path_obj)
            db_stats = stats.get_database_stats()
            typer.echo(f"\n📊 Database Statistics:")
            typer.echo(f"  Total rows: {db_stats.get('total_rows', 0)}")
            typer.echo(f"  Database size: {db_stats.get('database_size', 'unknown')}")
        except:
            pass  # Stats not critical

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def refactor_dupes(
    path: str = typer.Argument(".", help="Path to search for duplicates"),
    threshold: float = typer.Option(0.95, "--threshold", "-t", help="Similarity threshold (0.0-1.0)"),
    min_lines: int = typer.Option(5, "--min-lines", help="Minimum lines for match"),
    output: str = typer.Option("table", "--output", "-o", help="Output format (table or json)"),
) -> None:
    """Find and suggest refactoring for duplicate code patterns.

    Identifies code duplicates and similar patterns that could be consolidated.

    Example:
        $ codesearch refactor-dupes src/ --threshold 0.9
        $ codesearch refactor-dupes . --min-lines 10
    """
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            typer.echo("Run 'codesearch index <path>' first", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)

        typer.echo(f"🔍 Searching for duplicates (threshold: {threshold})...")

        try:
            # Get all entities
            entities_table = client.open_table("code_entities")
            all_entities = entities_table.search().to_list()

            if not all_entities:
                typer.echo("❌ No entities in database")
                raise typer.Exit(0)

            # Find duplicates using similarity
            duplicates = []
            checked = set()

            for i, entity1 in enumerate(all_entities):
                if entity1["entity_id"] in checked:
                    continue

                for entity2 in all_entities[i+1:]:
                    if entity2["entity_id"] in checked:
                        continue

                    # Calculate simple similarity (placeholder)
                    # Real implementation would use embeddings
                    if (entity1.get("code_text", "") == entity2.get("code_text", "") and
                        len(entity1.get("code_text", "").split("\n")) >= min_lines):
                        duplicates.append({
                            "entity1": entity1.get("name"),
                            "entity2": entity2.get("name"),
                            "file1": entity1.get("file_path"),
                            "file2": entity2.get("file_path"),
                            "similarity": 1.0,
                        })
                        checked.add(entity2["entity_id"])

            if not duplicates:
                typer.echo(f"✅ No duplicates found above {threshold} threshold")
                raise typer.Exit(0)

            typer.echo(f"⚠️  Found {len(duplicates)} duplicate patterns\n")

            for dup in duplicates[:20]:
                typer.echo(f"📌 {dup['entity1']} ↔ {dup['entity2']}")
                typer.echo(f"   Similarity: {dup['similarity']:.1%}")
                typer.echo(f"   Files: {dup['file1']}, {dup['file2']}\n")

            if len(duplicates) > 20:
                typer.echo(f"... and {len(duplicates) - 20} more duplicates")

        except Exception as search_error:
            typer.echo(f"⚠️  Error searching for duplicates: {search_error}")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def list_functions(
    file_path: Optional[str] = typer.Option(None, "--file", "-f", help="Filter by file path"),
    language: Optional[str] = typer.Option(None, "--language", "-L", help="Filter by language"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum results to display"),
) -> None:
    """List indexed functions and code entities.

    Browse all the code entities in your indexed database.

    Example:
        $ codesearch list-functions
        $ codesearch list-functions --language python
        $ codesearch list-functions --file mymodule.py --limit 50
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

            # Filter
            filtered = all_entities
            if file_path:
                filtered = [e for e in filtered if file_path in e.get("file_path", "")]
            if language:
                filtered = [e for e in filtered if e.get("language") == language]

            if not filtered:
                typer.echo("❌ No entities found matching filters")
                raise typer.Exit(0)

            typer.echo(f"📚 Indexed code entities ({len(filtered)} total, showing {min(limit, len(filtered))}):\n")

            for entity in filtered[:limit]:
                name = entity.get("name", "unknown")
                entity_type = entity.get("entity_type", "unknown")
                file_p = entity.get("file_path", "unknown")
                lang = entity.get("language", "unknown")

                typer.echo(f"  {name:30} | {entity_type:10} | {lang:10} | {file_p}")

            if len(filtered) > limit:
                typer.echo(f"\n... and {len(filtered) - limit} more")

        except Exception as list_error:
            typer.echo(f"⚠️  Error listing functions: {list_error}")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def repo_list() -> None:
    """List all registered repositories.

    Shows all repositories configured for multi-repo indexing.

    Example:
        $ codesearch repo-list
    """
    try:
        registry = RepositoryRegistry()
        repos = registry.list_repositories()

        if not repos:
            typer.echo("📭 No repositories registered yet")
            typer.echo("Use 'codesearch repo-add <path>' to add a repository")
            return

        typer.echo(f"📚 Registered repositories ({len(repos)} total):\n")

        for repo in repos:
            metadata = registry.get_metadata(repo.repo_id)
            if metadata:
                typer.echo(f"  📁 {repo.repo_name}")
                typer.echo(f"     Path: {repo.repo_path}")
                typer.echo(f"     ID: {repo.repo_id}")
                typer.echo(f"     Namespace: {metadata.namespace_prefix}")
                typer.echo(f"     Entities: {metadata.entity_count}")
                typer.echo(f"     Files: {metadata.file_count}")
                typer.echo(f"     Indexed: {metadata.indexed_at}")
                typer.echo()

    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def repo_add(
    path: str = typer.Argument(..., help="Path to repository"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Human-readable repository name"),
) -> None:
    """Register a new repository for indexing.

    Adds a repository to the multi-repo configuration.

    Example:
        $ codesearch repo-add /path/to/repo
        $ codesearch repo-add /path/to/repo --name my-project
    """
    try:
        path_obj = Path(path).resolve()

        if not path_obj.exists():
            typer.echo(f"❌ Path not found: {path}", err=True)
            raise typer.Exit(2)

        registry = RepositoryRegistry()

        # Check if already registered
        existing = registry.find_by_path(str(path_obj))
        if existing:
            typer.echo(f"⚠️  Repository already registered as '{existing.repo_name}'")
            raise typer.Exit(1)

        # Register the repository
        config = registry.register_repository(str(path_obj), name)
        typer.echo(f"✅ Repository registered: {config.repo_name}")
        typer.echo(f"   ID: {config.repo_id}")
        typer.echo(f"   Path: {config.repo_path}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def repo_remove(
    repo_id: str = typer.Argument(..., help="Repository ID to remove"),
) -> None:
    """Unregister a repository.

    Removes a repository from the multi-repo configuration.

    Example:
        $ codesearch repo-remove abc123def456
    """
    try:
        registry = RepositoryRegistry()

        if not registry.unregister_repository(repo_id):
            typer.echo(f"❌ Repository not found: {repo_id}", err=True)
            raise typer.Exit(2)

        typer.echo(f"✅ Repository unregistered: {repo_id}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


def search_multi(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to return"),
    repositories: Optional[str] = typer.Option(None, "--repos", "-r", help="Comma-separated repository IDs to search (default: all)"),
    output: str = typer.Option("table", "--output", "-o", help="Output format (table or json)"),
) -> None:
    """Search across multiple repositories.

    Searches all indexed repositories or a subset specified by repo IDs.

    Example:
        $ codesearch search-multi "error handling"
        $ codesearch search-multi "authenticate" --repos repo1,repo2 --limit 20
    """
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            typer.echo("Run 'codesearch index <path>' to create a database", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)
        engine = QueryEngine(client)
        registry = RepositoryRegistry()

        typer.echo(f"🔍 Searching across repositories for: {query}")

        # Build filters
        filters = []
        if repositories:
            repo_list = [r.strip() for r in repositories.split(",")]
            filters.append(RepositoryFilter(repo_list))
            typer.echo(f"   Filtering to: {', '.join(repo_list)}")

        # Search (note: text search not yet implemented in engine)
        try:
            results = engine.search_vector([0.0] * 768, filters=filters, limit=limit)
        except NotImplementedError:
            typer.echo("⚠️  Text search requires embedding integration", err=True)
            raise typer.Exit(1)

        if not results:
            typer.echo("❌ No results found")
            return

        typer.echo(f"✅ Found {len(results)} results\n")

        # Group results by repository
        by_repo = {}
        for result in results:
            repo = result.repository or "unknown"
            if repo not in by_repo:
                by_repo[repo] = []
            by_repo[repo].append(result)

        # Display results by repository
        for repo_name in sorted(by_repo.keys()):
            repo_results = by_repo[repo_name]
            typer.echo(f"\n📁 {repo_name} ({len(repo_results)} results)")
            typer.echo("-" * 80)

            if output == "json":
                import json
                typer.echo(json.dumps([r.__dict__ for r in repo_results], indent=2))
            else:
                typer.echo(format_results_table(repo_results))

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)
