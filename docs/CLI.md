# CLI Command Reference - Codesearch

Codesearch provides a powerful command-line interface for semantic code search. All commands follow standard Unix conventions with help available via `--help`.

## Global Options

All commands support these global options:

```bash
--help                Show command help
--version             Show Codesearch version
```

## Command Overview

| Command | Purpose | Status |
|---------|---------|--------|
| `pattern` | Semantic search by natural language | ✅ Working |
| `find-similar` | Find similar functions | ✅ Working |
| `index` | Index a repository | ⚠️ Stub |
| `dependencies` | Show call relationships | ⚠️ Stub |
| `refactor-dupes` | Find duplicate code | ⚠️ Stub |

---

## `codesearch pattern`

Search for code matching a natural language description.

### Usage

```bash
codesearch pattern <query> [OPTIONS]
```

### Arguments

- **`query`** (required): Natural language description of what you're searching for

### Options

```bash
--language LANG          Filter by programming language (python, typescript, go)
--limit N               Maximum results to return (default: 10)
--offset N              Skip first N results for pagination (default: 0)
--file PATTERN          Filter by file path pattern
--repo ID               Filter by repository ID
--threshold SCORE       Similarity threshold 0.0-1.0 (default: 0.0)
--format FORMAT         Output format: table or json (default: table)
```

### Configuration

Can be set via environment variables:
- `CODESEARCH_LANGUAGE` - Default language filter
- `CODESEARCH_OUTPUT_FORMAT` - Default output format (table|json)

### Examples

**Basic semantic search:**
```bash
codesearch pattern "function that validates email addresses"
```

**Search for error handling code:**
```bash
codesearch pattern "catches and handles exceptions"
```

**Find database queries with filtering:**
```bash
codesearch pattern "database query" --language python --limit 20
```

**Search with similarity threshold:**
```bash
codesearch pattern "authentication" --threshold 0.85
```

**Get results as JSON for programmatic use:**
```bash
codesearch pattern "caching" --format json | jq '.results[].name'
```

**Pagination through large result sets:**
```bash
codesearch pattern "validation" --limit 5 --offset 0
codesearch pattern "validation" --limit 5 --offset 5
```

### Output

**Table Format (default):**
```
╭────────────────────────────────────────────────────────────────────╮
│                           Code Entities                            │
├────────────────────┬──────────┬──────────────┬─────────────────────┤
│ Name               │ Type     │ Language     │ File                │
├────────────────────┼──────────┼──────────────┼─────────────────────┤
│ validate_email     │ function │ python       │ utils/validators.py │
│ is_valid_email     │ function │ python       │ auth/email.py       │
│ check_email_format │ function │ python       │ models/user.py      │
╰────────────────────┴──────────┴──────────────┴─────────────────────╯
```

**JSON Format:**
```json
{
  "count": 3,
  "total": 15,
  "results": [
    {
      "id": "abc123",
      "name": "validate_email",
      "type": "function",
      "language": "python",
      "file": "utils/validators.py",
      "line": 42,
      "signature": "def validate_email(email: str) -> bool",
      "score": 0.92
    }
  ]
}
```

### Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `Database not found` | No index exists | Run `codesearch index <path>` first |
| `Invalid query` | Empty or malformed query | Provide a non-empty description |
| `Language not supported` | Invalid language filter | Use: python, typescript, or go |
| `No results found` | No matching code | Try broader search terms |

---

## `codesearch find-similar`

Find functions similar to a given function by name.

### Usage

```bash
codesearch find-similar <entity_name> [OPTIONS]
```

### Arguments

- **`entity_name`** (required): Name of the function/class to find similar code for

### Options

```bash
--language LANG         Filter results by language (python, typescript, go)
--limit N              Maximum results to return (default: 10)
--offset N             Skip first N results (default: 0)
--threshold SCORE      Similarity threshold 0.0-1.0 (default: 0.0)
--format FORMAT        Output format: table or json (default: table)
--include-self        Include the original entity in results (default: false)
```

### Examples

**Find similar functions:**
```bash
codesearch find-similar validate_input
```

**Find similar with specific language:**
```bash
codesearch find-similar DatabaseConnection --language typescript
```

**Strict similarity matching:**
```bash
codesearch find-similar error_handler --threshold 0.90
```

**Include the original function in results:**
```bash
codesearch find-similar my_function --include-self
```

**Export as JSON for further processing:**
```bash
codesearch find-similar api_call --format json > similar_functions.json
```

### Output

Same as `pattern` command - table or JSON format depending on `--format` option.

### Use Cases

1. **Find Code Duplicates**: Find similar implementations that could be consolidated
2. **Discover Patterns**: Find different implementations of the same concept
3. **Code Review**: See how similar functionality is implemented elsewhere
4. **Refactoring**: Identify functions suitable for abstraction

---

## `codesearch dependencies`

Show call relationships for a function (what calls it, what it calls).

### Usage

```bash
codesearch dependencies <entity_name> [OPTIONS]
```

### Arguments

- **`entity_name`** (required): Name of the function/class to analyze

### Options

```bash
--direction DIRECTION   Show: callers, callees, or both (default: both)
--depth N              Recursion depth (default: 1)
--format FORMAT        Output format: tree or json (default: tree)
--language LANG        Filter results by language
```

### Examples

**Show all call relationships:**
```bash
codesearch dependencies process_request
```

**Show only callers (functions that call this one):**
```bash
codesearch dependencies handle_error --direction callers
```

**Show only callees (functions called by this one):**
```bash
codesearch dependencies main --direction callees
```

**Deep dependency analysis (2 levels):**
```bash
codesearch dependencies api_handler --depth 2
```

**Export dependency graph as JSON:**
```bash
codesearch dependencies database_query --format json
```

### Output

**Tree Format (default):**
```
process_request
├── callers:
│   ├── main (main.py:10)
│   └── async_worker (tasks.py:45)
└── callees:
    ├── validate_input (validators.py:20)
    ├── query_database (db.py:100)
    └── log_result (logging.py:55)
```

**JSON Format:**
```json
{
  "entity": "process_request",
  "callers": [
    {"name": "main", "file": "main.py", "line": 10},
    {"name": "async_worker", "file": "tasks.py", "line": 45}
  ],
  "callees": [
    {"name": "validate_input", "file": "validators.py", "line": 20},
    {"name": "query_database", "file": "db.py", "line": 100}
  ]
}
```

### Status

⚠️ **Currently a stub** - Requires Query Infrastructure (Issue #11) implementation.

---

## `codesearch index`

Index a repository or directory for semantic search.

### Usage

```bash
codesearch index <path> [OPTIONS]
```

### Arguments

- **`path`** (required): Path to repository or directory to index

### Options

```bash
--database PATH        Database path (default: CODESEARCH_DB_PATH or ~/.codesearch)
--language LANG        Index specific language (python, typescript, go, or all)
--incremental         Only index changed files (default: true)
--full-reindex        Force complete re-indexing
--model MODEL         Embedding model to use
--batch-size N        Batch size for embedding (default: 32)
--exclude PATTERN     Exclude file patterns (e.g., *.test.py)
--workers N           Number of parallel workers (default: CPU count)
--repo-id ID          Repository identifier for multi-repo support
```

### Examples

**Simple indexing:**
```bash
codesearch index ~/my-project
```

**Index only Python files:**
```bash
codesearch index ~/my-project --language python
```

**Force full re-indexing:**
```bash
codesearch index ~/my-project --full-reindex
```

**Index with custom database location:**
```bash
codesearch index ~/my-project --database /custom/path/.codesearch
```

**Index multiple repositories with different IDs:**
```bash
codesearch index ~/repo-a --repo-id repo-a
codesearch index ~/repo-b --repo-id repo-b
```

**Exclude test files:**
```bash
codesearch index ~/my-project --exclude '*.test.py' --exclude '*.spec.ts'
```

**Parallel indexing with custom workers:**
```bash
codesearch index ~/large-project --workers 8
```

### Output

```
Indexing ~/my-project...
✓ Discovered 1,250 files
✓ Found 3,456 code entities
✓ Generated embeddings for 3,456 entities
✓ Successfully indexed 3,456 entities
✓ Failed: 2 (parse errors)
✓ Skipped: 45 (no changes)

Index complete. Database: ~/.codesearch (42.5 MB)
```

### Status

⚠️ **Currently a stub** - Requires Data Ingestion Pipeline (Issue #10) implementation.

---

## `codesearch refactor-dupes`

Find duplicate or highly similar code that could be consolidated.

### Usage

```bash
codesearch refactor-dupes [OPTIONS]
```

### Options

```bash
--threshold SCORE     Similarity threshold for duplicates (default: 0.90)
--language LANG       Check specific language (python, typescript, go, or all)
--group-by TYPE       Group results by: location or similarity (default: similarity)
--min-size N          Minimum lines of code to consider (default: 3)
--exclude PATTERN     Exclude file patterns
--format FORMAT       Output format: table, json, or report (default: table)
```

### Examples

**Find obvious duplicates:**
```bash
codesearch refactor-dupes --threshold 0.95
```

**Find similar functions (more lenient):**
```bash
codesearch refactor-dupes --threshold 0.80
```

**Find Python duplicates:**
```bash
codesearch refactor-dupes --language python
```

**Find significant duplicates (5+ lines):**
```bash
codesearch refactor-dupes --min-size 5
```

**Generate refactoring report:**
```bash
codesearch refactor-dupes --format report
```

### Output

**Table Format:**
```
╭─────────────────────────────────────────────────────────────────────╮
│                      Duplicate Code Patterns                        │
├─────────────────────┬─────────────┬──────────────┬──────────────────┤
│ Function Name       │ Location 1  │ Location 2   │ Similarity Score │
├─────────────────────┼─────────────┼──────────────┼──────────────────┤
│ validate_email      │ auth.py:42  │ utils.py:100 │ 0.95             │
│ parse_json          │ api.py:20   │ models.py:15 │ 0.92             │
│ format_output       │ cli.py:40   │ web.py:28    │ 0.88             │
╰─────────────────────┴─────────────┴──────────────┴──────────────────╯
```

**Report Format:**
```
DUPLICATE CODE ANALYSIS REPORT
==============================

Total Duplicates Found: 12
Total Code Duplication: 2,340 lines
Estimated Consolidation Potential: 30%

TOP CANDIDATES FOR REFACTORING:
1. validate_email (0.95 similarity)
   - auth.py:42-65 (24 lines)
   - utils.py:100-123 (24 lines)
   Estimated savings: 24 lines

2. parse_json (0.92 similarity)
   - api.py:20-35 (16 lines)
   - models.py:15-30 (16 lines)
   Estimated savings: 16 lines
```

### Status

⚠️ **Currently a stub** - Requires full Query Infrastructure implementation.

---

## Configuration

### Environment Variables

Control default behavior via environment variables:

```bash
# Database location
export CODESEARCH_DB_PATH=~/.codesearch

# Default output format
export CODESEARCH_OUTPUT_FORMAT=table

# Default language filter
export CODESEARCH_LANGUAGE=python
```

### Configuration File (Planned)

Future versions will support `~/.codesearch/config.yaml`:

```yaml
database:
  path: ~/.codesearch
  embeddings_model: sentence-transformers/all-MiniLM-L6-v2

cli:
  default_language: python
  default_format: table
  default_limit: 10

indexing:
  batch_size: 32
  workers: 4
  exclude_patterns:
    - "*.test.py"
    - "*.spec.ts"
```

---

## Output Formats

### Table Format

Beautiful terminal output with colors and alignment:
- Default for interactive terminal use
- Automatically adapts to terminal width
- Supports scrolling for large result sets

### JSON Format

Machine-readable output:
- Perfect for scripts and automation
- Parse with `jq` for advanced filtering
- Preserve all metadata

Example:
```bash
codesearch pattern "validation" --format json | jq '.results | length'
```

---

## Exit Codes

All commands use standard exit codes:

| Code | Meaning | Example |
|------|---------|---------|
| 0 | Success | Command completed successfully |
| 1 | Configuration Error | Invalid arguments or environment |
| 2 | Database Error | Database connection or schema issue |
| 3 | Query Error | Invalid query or entity not found |

### Examples

```bash
# Check exit code
codesearch pattern "validation"
echo $?  # Prints 0 on success

# Conditional based on exit code
codesearch find-similar my_function && echo "Found" || echo "Not found"
```

---

## Advanced Usage

### Piping Results

Combine with standard Unix tools:

```bash
# Count results
codesearch pattern "database" --format json | jq '.count'

# Filter results
codesearch pattern "validation" --format json | \
  jq '.results[] | select(.language=="python")'

# Extract entity names
codesearch find-similar api --format json | \
  jq -r '.results[].name'
```

### Scripting

Create reusable search scripts:

```bash
#!/bin/bash
# search-by-pattern.sh

PATTERN="$1"
LANGUAGE="${2:-python}"
LIMIT="${3:-20}"

codesearch pattern "$PATTERN" \
  --language "$LANGUAGE" \
  --limit "$LIMIT" \
  --format json
```

Usage:
```bash
./search-by-pattern.sh "error handling" python 15
```

---

## Troubleshooting

### Common Issues

**Q: "Database not found" error**
- A: Run `codesearch index /path/to/repo` first

**Q: No results for valid queries**
- A: Check if database has been indexed with relevant language

**Q: Commands are slow**
- A: Ensure database path is on fast storage (SSD recommended)

**Q: Help text not displaying**
- A: Run `codesearch --help` or `codesearch <command> --help`

For more troubleshooting, see [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## See Also

- [Architecture](ARCHITECTURE.md) - System design
- [Python API](API.md) - Programmatic usage
- [Contributing](../CONTRIBUTING.md) - Development guide
