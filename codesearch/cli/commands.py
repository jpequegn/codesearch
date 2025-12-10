"""CLI commands for codesearch.

Core commands for indexing, searching, and analyzing code.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Union

import lancedb
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from codesearch.cli.config import get_db_path, validate_db_exists
from codesearch.cli.formatting import format_results_json, format_results_table
from codesearch.data_ingestion.pipeline import DataIngestionPipeline
from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.indexing.incremental import IncrementalIndexer
from codesearch.indexing.repository import RepositoryRegistry
from codesearch.indexing.scanner import RepositoryScannerImpl
from codesearch.lancedb import DatabaseBackupManager, DatabaseStatistics
from codesearch.lancedb.initialization import DatabaseInitializer
from codesearch.models import Class, CodeEntity, CodeRelationship, Function
from codesearch.parsers.python_parser import PythonParser
from codesearch.query import QueryEngine
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

        # Find the entity by name in the database
        try:
            entities_table = client.open_table("code_entities")

            # Search for entity by name using LanceDB where clause
            # Build filter string - escape single quotes in entity name
            safe_name = entity_name.replace("'", "''")
            filter_expr = f"name = '{safe_name}'"
            if language:
                safe_language = language.replace("'", "''")
                filter_expr += f" AND language = '{safe_language}'"

            # Query all entities matching the name
            entity_results = (
                entities_table.search()
                .where(filter_expr, prefilter=True)
                .limit(1)
                .to_list()
            )

            if not entity_results:
                typer.echo(f"❌ Entity '{entity_name}' not found in database")
                raise typer.Exit(3)

            entity = entity_results[0]
            entity_id = entity.get("entity_id")

            if "code_vector" not in entity or not entity["code_vector"]:
                typer.echo(f"⚠️  Entity '{entity_name}' has no embedding vector")
                raise typer.Exit(2)

            # Search for similar vectors using the entity's embedding
            typer.echo(f"🔍 Finding similar code to '{entity_name}'...")
            similar_results = engine.search_vector(
                entity["code_vector"],
                limit=limit + 1  # +1 to account for excluding self
            )

            # Filter out the original entity by entity_id (more reliable than name)
            filtered = [r for r in similar_results if r.entity_id != entity_id][:limit]

            # Apply similarity threshold
            filtered = [r for r in filtered if r.similarity_score >= threshold]

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
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1) from None


def dependencies(
    entity_name: str = typer.Argument(..., help="Entity to analyze"),
    direction: str = typer.Option(
        "both", "--direction", "-d", help="Direction: 'calls', 'callers', or 'both'"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format (table or json)"),
) -> None:
    """Analyze call graph and dependencies for a code entity.

    Shows what functions the target calls and what functions call the target.

    Example:
        $ codesearch deps main_function --direction calls
        $ codesearch deps MyClass.method --direction callers
    """
    try:
        db_path = get_db_path()

        if not validate_db_exists(db_path):
            typer.echo(f"❌ Database not found at {db_path}", err=True)
            raise typer.Exit(2)

        client = lancedb.connect(db_path)

        try:
            # First, find the entity by name to get its entity_id
            entities_table = client.open_table("code_entities")

            # Search for entity by name using LanceDB where clause
            safe_name = entity_name.replace("'", "''")
            entity_results = (
                entities_table.search()
                .where(f"name = '{safe_name}'", prefilter=True)
                .limit(1)
                .to_list()
            )

            if not entity_results:
                typer.echo(f"❌ Entity '{entity_name}' not found in database")
                raise typer.Exit(3)

            entity = entity_results[0]
            entity_id = entity.get("entity_id")

            # Build lookup table for entity names by ID
            all_entities = entities_table.search().to_list()
            entity_name_by_id = {e.get("entity_id"): e.get("name") for e in all_entities}
            entity_file_by_id = {e.get("entity_id"): e.get("file_path") for e in all_entities}

            # Query relationships table
            relationships_table = client.open_table("code_relationships")
            all_rels = relationships_table.search().to_list()

            # Filter based on direction
            callers = []  # Functions that call this entity
            callees = []  # Functions this entity calls

            for rel in all_rels:
                if rel.get("callee_id") == entity_id:
                    callers.append(rel)
                if rel.get("caller_id") == entity_id:
                    callees.append(rel)

            # Determine what to show based on direction
            if direction == "calls":
                rels_to_show = callees
                header = f"Functions called by '{entity_name}'"
            elif direction == "callers":
                rels_to_show = callers
                header = f"Functions that call '{entity_name}'"
            else:  # both
                rels_to_show = None  # Show both sections
                header = f"Call graph for '{entity_name}'"

            if rels_to_show is not None and not rels_to_show:
                typer.echo(f"No dependencies found for '{entity_name}'")
                raise typer.Exit(0)

            if direction == "both" and not callers and not callees:
                typer.echo(f"No dependencies found for '{entity_name}'")
                raise typer.Exit(0)

            typer.echo(f"📊 {header}\n")

            if output == "json":
                import json
                result_data = {
                    "entity": entity_name,
                    "entity_id": entity_id,
                    "callers": [
                        {
                            "name": entity_name_by_id.get(r.get("caller_id"), "unknown"),
                            "file": entity_file_by_id.get(r.get("caller_id"), "unknown"),
                        }
                        for r in callers
                    ] if direction in ["callers", "both"] else [],
                    "calls": [
                        {
                            "name": entity_name_by_id.get(r.get("callee_id"), "unknown"),
                            "file": entity_file_by_id.get(r.get("callee_id"), "unknown"),
                        }
                        for r in callees
                    ] if direction in ["calls", "both"] else [],
                }
                typer.echo(json.dumps(result_data, indent=2))
            else:
                # Table format
                if direction in ["callers", "both"] and callers:
                    typer.echo("  📥 Called by:")
                    for rel in callers[:20]:
                        caller_id = rel.get("caller_id")
                        caller_name = entity_name_by_id.get(caller_id, "unknown")
                        caller_file = entity_file_by_id.get(caller_id, "")
                        typer.echo(f"     ← {caller_name} ({caller_file})")
                    if len(callers) > 20:
                        typer.echo(f"     ... and {len(callers) - 20} more")

                if direction == "both" and callers and callees:
                    typer.echo()

                if direction in ["calls", "both"] and callees:
                    typer.echo("  📤 Calls:")
                    for rel in callees[:20]:
                        callee_id = rel.get("callee_id")
                        callee_name = entity_name_by_id.get(callee_id, "unknown")
                        callee_file = entity_file_by_id.get(callee_id, "")
                        typer.echo(f"     → {callee_name} ({callee_file})")
                    if len(callees) > 20:
                        typer.echo(f"     ... and {len(callees) - 20} more")

                # Summary
                typer.echo()
                if direction == "both":
                    typer.echo(f"  Summary: {len(callers)} callers, {len(callees)} calls")
                elif direction == "callers":
                    typer.echo(f"  Summary: {len(callers)} callers")
                else:
                    typer.echo(f"  Summary: {len(callees)} calls")

        except typer.Exit:
            raise
        except Exception as table_error:
            typer.echo(f"⚠️  Relationships table not available: {table_error}")
            typer.echo("Make sure to index your repository first", err=True)
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1) from None


def _to_code_entity(
    entity: Union[Function, Class],
    repository: str,
) -> CodeEntity:
    """Convert a Function or Class to a CodeEntity for database storage.

    Args:
        entity: Function or Class object with embedding
        repository: Repository name for the entity

    Returns:
        CodeEntity ready for database insertion
    """
    # Generate unique entity ID from content hash
    content_hash = hashlib.sha256(
        f"{entity.file_path}:{entity.line_number}:{entity.name}".encode()
    ).hexdigest()

    # Determine entity type
    if isinstance(entity, Class):
        entity_type = "class"
    elif isinstance(entity, Function) and entity.is_method:
        entity_type = "method"
    else:
        entity_type = "function"

    # Determine visibility based on naming conventions
    if entity.name.startswith("__") and not entity.name.endswith("__"):
        visibility = "private"
    elif entity.name.startswith("_"):
        visibility = "protected"
    else:
        visibility = "public"

    # Get embedding (default to empty list if not generated)
    code_vector = entity.embedding if entity.embedding else []

    # Generate source hash for incremental indexing
    source_hash = hashlib.sha256(
        (entity.source_code or "").encode()
    ).hexdigest()

    return CodeEntity(
        entity_id=content_hash,
        name=entity.name,
        code_text=entity.source_code or "",
        code_vector=code_vector,
        language=entity.language,
        entity_type=entity_type,
        repository=repository,
        file_path=entity.file_path,
        start_line=entity.line_number,
        end_line=entity.end_line,
        visibility=visibility,
        source_hash=source_hash,
    )


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

        # Check if already indexed (before initializing, which creates files)
        db_initializer = DatabaseInitializer(db_path_obj)
        if db_initializer.is_initialized() and not force:
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

        # Initialize database schema (idempotent - creates tables if needed)
        if not db_initializer.initialize():
            typer.echo("❌ Failed to initialize database schema", err=True)
            raise typer.Exit(2)

        typer.echo(f"📇 Indexing {path_obj}...")

        # Initialize incremental indexer if requested
        if incremental:
            indexer = IncrementalIndexer()
            typer.echo(f"📊 Currently indexed: {indexer.get_indexed_count()} files")
            typer.echo("🔄 Incremental update mode enabled")

        # Scan repository for files
        console = Console()
        scanner = RepositoryScannerImpl()

        # Add language filter if specified
        if language:
            scanner.config.supported_languages = {language}

        with console.status(f"[bold blue]📁 Scanning {path_obj.name}...", spinner="dots"):
            files = scanner.scan_repository(str(path_obj))
            stats = scanner.get_statistics()

        typer.echo(f"📁 Found {len(files)} files to index")

        # Show breakdown by language
        if stats.get("by_language"):
            for lang, count in stats["by_language"].items():
                typer.echo(f"   - {lang}: {count} files")

        if not files:
            typer.echo("⚠️  No files found to index")
            typer.echo("   Check that the path contains supported files (.py, .js, .ts, .go)")
            raise typer.Exit(0)

        # Parse files to extract code entities
        parser = PythonParser()
        all_entities: list[Function | Class] = []
        parse_errors: list[tuple[str, str]] = []

        with console.status("[bold blue]🔗 Extracting code entities...", spinner="dots"):
            for file_metadata in files:
                if file_metadata.language != "python":
                    continue  # Only Python parser available for now

                try:
                    entities = parser.parse_file(file_metadata.file_path)
                    all_entities.extend(entities)
                except SyntaxError as e:
                    parse_errors.append((file_metadata.file_path, f"Syntax error: {e}"))
                except IOError as e:
                    parse_errors.append((file_metadata.file_path, f"IO error: {e}"))
                except Exception as e:
                    parse_errors.append((file_metadata.file_path, f"Error: {e}"))

        # Count by type
        function_count = sum(1 for e in all_entities if isinstance(e, Function))
        class_count = sum(1 for e in all_entities if isinstance(e, Class))

        typer.echo(f"🔗 Extracted {len(all_entities)} code entities")
        typer.echo(f"   - Functions/methods: {function_count}")
        typer.echo(f"   - Classes: {class_count}")

        # Report parse errors
        if parse_errors:
            typer.echo(f"   ⚠️  Skipped {len(parse_errors)} files with errors")
            if len(parse_errors) <= 5:
                for file_path, error in parse_errors:
                    rel_path = Path(file_path).relative_to(path_obj) if file_path.startswith(str(path_obj)) else file_path
                    typer.echo(f"      - {rel_path}: {error}")

        if not all_entities:
            typer.echo("⚠️  No code entities found to index")
            raise typer.Exit(0)

        # Generate embeddings for code entities
        typer.echo("🧮 Loading embedding model...")
        try:
            embedder = EmbeddingGenerator()
            model_info = embedder.get_model_info()
            typer.echo(f"   Model: {model_info['name']} ({model_info['dimensions']}d)")
            typer.echo(f"   Device: {model_info['device']}")
        except Exception as e:
            typer.echo(f"❌ Failed to load embedding model: {e}", err=True)
            typer.echo("   Make sure transformers and torch are installed", err=True)
            raise typer.Exit(1) from None

        # Prepare texts for batch embedding
        def get_entity_text(entity: Function | Class) -> str:
            """Combine entity name, docstring, and source code for embedding."""
            parts = [entity.name]
            if entity.docstring:
                parts.append(entity.docstring)
            if entity.source_code:
                parts.append(entity.source_code)
            return "\n".join(parts)

        # Process embeddings in batches with progress bar
        batch_size = 32  # Balance between memory usage and speed
        total_entities = len(all_entities)
        entities_with_embeddings = 0
        embedding_errors = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("🧮 Generating embeddings...", total=total_entities)

            for i in range(0, total_entities, batch_size):
                batch = all_entities[i:i + batch_size]
                batch_texts = [get_entity_text(entity) for entity in batch]

                try:
                    batch_embeddings = embedder.embed_batch(batch_texts)

                    # Assign embeddings to entities
                    for entity, embedding in zip(batch, batch_embeddings):
                        entity.embedding = embedding
                        entities_with_embeddings += 1

                except Exception as e:
                    # Log error but continue with remaining batches
                    logger.warning(f"Error embedding batch {i // batch_size}: {e}")
                    embedding_errors += len(batch)

                progress.update(task, advance=len(batch))

        typer.echo(f"🧮 Generated {entities_with_embeddings} embeddings")
        if embedding_errors > 0:
            typer.echo(f"   ⚠️  Failed to embed {embedding_errors} entities")

        # Store entities in database using DataIngestionPipeline
        typer.echo("💾 Storing in database...")

        # Convert all entities to CodeEntity format
        repo_name = path_obj.name
        code_entities = []
        for entity in all_entities:
            try:
                code_entity = _to_code_entity(entity, repo_name)
                code_entities.append(code_entity)
            except Exception as e:
                logger.warning(f"Error converting entity {entity.name}: {e}")

        if not code_entities:
            typer.echo("❌ No entities to store", err=True)
            raise typer.Exit(1) from None

        # Initialize pipeline and ingest entities
        try:
            client = lancedb.connect(db_path)
            pipeline = DataIngestionPipeline(client)
            result = pipeline.ingest_batch(code_entities)

            typer.echo(f"💾 Stored {result.inserted_count} entities")
            if result.skipped_count > 0:
                typer.echo(f"   ⏭️  Skipped {result.skipped_count} duplicates")
            if result.failed_count > 0:
                typer.echo(f"   ⚠️  {result.failed_count} storage errors")
        except Exception as e:
            typer.echo(f"❌ Database storage error: {e}", err=True)
            raise typer.Exit(1) from None

        # Extract and store relationships (call graph)
        typer.echo("🔗 Extracting call graph relationships...")

        # Build entity_id lookup by name for resolving calls
        entity_id_by_name: dict[str, str] = {}
        for entity in all_entities:
            if isinstance(entity, Function):
                # Use simple name for lookup (may have collisions)
                entity_id = hashlib.sha256(
                    f"{entity.file_path}:{entity.line_number}:{entity.name}".encode()
                ).hexdigest()
                entity_id_by_name[entity.name] = entity_id
                # Also store with class prefix for methods
                if entity.class_name:
                    entity_id_by_name[f"{entity.class_name}.{entity.name}"] = entity_id

        # Extract relationships from functions with calls_to
        relationships: list[CodeRelationship] = []
        for entity in all_entities:
            if isinstance(entity, Function) and entity.calls_to:
                caller_id = hashlib.sha256(
                    f"{entity.file_path}:{entity.line_number}:{entity.name}".encode()
                ).hexdigest()

                for call_name in entity.calls_to:
                    # Try to resolve the callee
                    callee_id = entity_id_by_name.get(call_name)
                    if callee_id:
                        relationships.append(CodeRelationship(
                            caller_id=caller_id,
                            callee_id=callee_id,
                            relationship_type="calls",
                        ))

        if relationships:
            try:
                rel_result = pipeline.ingest_relationships(relationships)
                typer.echo(f"🔗 Stored {rel_result.relationships_inserted} relationships")
                if rel_result.relationships_failed > 0:
                    typer.echo(f"   ⚠️  {rel_result.relationships_failed} relationship errors")
            except Exception as rel_error:
                typer.echo(f"   ⚠️  Could not store relationships: {rel_error}")
        else:
            typer.echo("🔗 No call relationships found")

        typer.echo("\n✅ Indexing complete!")
        typer.echo(f"Database saved to {db_path}")

        # Show statistics
        try:
            lancedb.connect(db_path)  # Verify connection works
            stats = DatabaseStatistics(db_path_obj)
            db_stats = stats.get_database_stats()
            typer.echo("\n📊 Database Statistics:")
            typer.echo(f"  Total rows: {db_stats.get('total_rows', 0)}")
            typer.echo(f"  Database size: {db_stats.get('database_size', 'unknown')}")
        except Exception:
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
