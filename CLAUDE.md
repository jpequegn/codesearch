# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install (dev mode with all dependencies)
pip install -e ".[dev]"

# Run tests
pytest                                    # All tests
pytest tests/cli/test_commands.py         # Specific file
pytest -k "test_pattern"                  # By name pattern
pytest --cov=codesearch --cov-report=html # With coverage

# Code quality
black codesearch/ tests/                  # Format
isort codesearch/ tests/                  # Sort imports
mypy codesearch/                          # Type check
ruff check codesearch/ tests/             # Lint

# CLI usage
codesearch --version
codesearch search "query"                 # Semantic search
codesearch find-similar <entity_name>     # Find similar code
codesearch deps <entity_name>             # Show dependencies
codesearch index /path/to/repo            # Index repository
codesearch refactor-dupes --threshold 0.85
```

## Architecture Overview

5-component semantic code search system using vector embeddings and LanceDB:

```
CLI (Typer) → Query Engine → LanceDB (vectors) + Call Graph → Embedding Pipeline → Code Parsers (tree-sitter)
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `codesearch/cli/` | Typer-based CLI with Rich formatting. Entry: `main.py`, commands in `commands.py` |
| `codesearch/query/` | QueryEngine, SearchResult, MetadataFilter for semantic search |
| `codesearch/lancedb/` | Vector DB layer. Schema: `code_entities`, `code_relationships`, `search_metadata` |
| `codesearch/embeddings/` | Transformer models (MiniLM 384-dim default, MPNet 768-dim) |
| `codesearch/indexing/` | Incremental indexing with hash-based deduplication |
| `codesearch/parsers/` | tree-sitter based parsing. Python fully implemented, TS/Go extensible |
| `codesearch/caching/` | AST and embedding caches with TTL |
| `codesearch/models.py` | Pydantic data models |

### LanceDB Schema

**code_entities**: entity_id (SHA256), name, type, signature, docstring, source_file, start/end_line, embedding (vector), language, repository_id, timestamps

**code_relationships**: relationship_id, caller_id, callee_id, call_count, call_locations, relationship_type, repository_id

**search_metadata**: entity_id, tags, complexity_score, line_count, test_coverage, last_modified

### Data Flow

**Indexing**: File Discovery → Hash Dedup → tree-sitter Parse → Entity Extraction → Embedding Generation → Validation → LanceDB Insert → Audit Trail

**Search**: Query → Embed Query → Vector Similarity Search → Metadata Filter → Pagination → Format Results

## Code Standards

- **Line length**: 100 chars (black/isort/ruff)
- **Type hints**: Required everywhere (mypy py3.9)
- **Docstrings**: Google-style
- **Naming**: snake_case functions, PascalCase classes, UPPER_SNAKE constants
- **Ruff rules**: E, F, I, N, W, ANN, B, C4, SIM (ignores E501, ANN101, ANN102)

## Environment Variables

```bash
CODESEARCH_DB_PATH=~/.codesearch     # Database location
CODESEARCH_LANGUAGE=python           # Default language filter
CODESEARCH_OUTPUT_FORMAT=table       # Output: table, json
```

## Exit Codes

0=success, 1=config error, 2=db error, 3=query error
