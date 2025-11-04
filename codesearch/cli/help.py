"""Help documentation and utilities for the CLI.

Provides extended help texts, command guides, and usage examples.
"""

import typer
from typing import Optional


def show_help() -> None:
    """Display comprehensive help information."""
    help_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          CODESEARCH - Help Guide                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Semantic code search using vector embeddings and LanceDB.

USAGE:
  codesearch [OPTIONS] COMMAND [ARGS]

COMMANDS:
  search            Search for code matching a semantic description
  find-similar      Find functions similar to a given entity
  deps              Analyze code dependencies
  index             Index a repository for searching
  refactor-dupes    Find and refactor duplicate code patterns

OPTIONS:
  --version, -v     Show version and exit
  --help, -h        Show this help message and exit

EXAMPLES:

  Search for code:
    $ codesearch search "error handling function"
    $ codesearch search "API authentication" --output json

  Find similar code:
    $ codesearch find-similar my_function --language python
    $ codesearch find-similar RequestHandler --limit 5

  Analyze dependencies:
    $ codesearch deps mymodule.py
    $ codesearch deps --recursive src/

  Index a repository:
    $ codesearch index /path/to/repository
    $ codesearch index . --language python

  Find duplicate patterns:
    $ codesearch refactor-dupes src/
    $ codesearch refactor-dupes --min-complexity 5

CONFIGURATION:

Configuration can be set via:
  1. Command-line arguments (highest priority)
  2. Environment variables (CODESEARCH_*)
  3. Configuration files (~/.codesearch/config.json)
  4. Default values (lowest priority)

Environment Variables:
  CODESEARCH_DB           Path to LanceDB database
  CODESEARCH_LANGUAGE     Default programming language
  CODESEARCH_OUTPUT       Default output format (table/json)
  CODESEARCH_LIMIT        Default result limit
  CODESEARCH_VERBOSE      Enable verbose output (true/false)

Configuration File Format (~/.codesearch/config.json):
  {
    "db_path": "/path/to/db",
    "output": "table",
    "limit": 10,
    "language": "python",
    "verbose": false
  }

For command-specific help:
  $ codesearch search --help
  $ codesearch index --help
  $ codesearch find-similar --help

DOCUMENTATION:
  Full documentation: https://github.com/jpequegn/codesearch
  Issue tracker: https://github.com/jpequegn/codesearch/issues
"""
    typer.echo(help_text)


def show_command_help(command: str) -> None:
    """Display help for a specific command.

    Args:
        command: Command name to show help for
    """
    help_texts = {
        "search": """
COMMAND: search

Search for code matching a semantic description.

USAGE:
  codesearch search QUERY [OPTIONS]

ARGUMENTS:
  QUERY                 Semantic search query (e.g., "error handler")

OPTIONS:
  --limit, -l INTEGER   Maximum number of results (default: 10)
  --output, -o TEXT     Output format: table or json (default: table)
  --language, -L TEXT   Filter results by language
  --help                Show this help message

EXAMPLES:
  $ codesearch search "authentication" --limit 5
  $ codesearch search "database connection" --output json
  $ codesearch search "error handling" --language python
""",
        "find-similar": """
COMMAND: find-similar

Find functions or classes similar to a given entity.

USAGE:
  codesearch find-similar ENTITY_NAME [OPTIONS]

ARGUMENTS:
  ENTITY_NAME           Name of the entity to match

OPTIONS:
  --limit, -l INTEGER   Maximum number of results (default: 10)
  --language, -L TEXT   Filter results by language
  --help                Show this help message

EXAMPLES:
  $ codesearch find-similar RequestHandler
  $ codesearch find-similar parse_json --language python
  $ codesearch find-similar --limit 20 db_connection
""",
        "deps": """
COMMAND: deps

Analyze code dependencies and call graphs.

USAGE:
  codesearch deps [FILE_PATH] [OPTIONS]

ARGUMENTS:
  FILE_PATH             Path to file or directory (default: current)

OPTIONS:
  --recursive, -r       Recursively analyze dependencies
  --format, -f TEXT     Output format: tree or json
  --help                Show this help message

EXAMPLES:
  $ codesearch deps mymodule.py
  $ codesearch deps src/ --recursive
  $ codesearch deps --format json
""",
        "index": """
COMMAND: index

Index a repository for semantic search.

USAGE:
  codesearch index [PATH] [OPTIONS]

ARGUMENTS:
  PATH                  Repository path to index (default: current)

OPTIONS:
  --language, -L TEXT   Programming language (auto-detect by default)
  --force, -f           Force reindex even if exists
  --help                Show this help message

EXAMPLES:
  $ codesearch index /path/to/repo
  $ codesearch index . --language python
  $ codesearch index --force
""",
        "refactor-dupes": """
COMMAND: refactor-dupes

Find and suggest refactoring for duplicate code patterns.

USAGE:
  codesearch refactor-dupes [PATH] [OPTIONS]

ARGUMENTS:
  PATH                  Repository path to analyze (default: current)

OPTIONS:
  --min-complexity INT  Minimum complexity to report (default: 0)
  --limit, -l INTEGER   Maximum results (default: 10)
  --output, -o TEXT     Output format: table or json
  --help                Show this help message

EXAMPLES:
  $ codesearch refactor-dupes src/
  $ codesearch refactor-dupes --min-complexity 5
  $ codesearch refactor-dupes --output json
""",
    }

    if command in help_texts:
        typer.echo(help_texts[command])
    else:
        typer.echo(f"No help available for command '{command}'")
        typer.echo("Run 'codesearch --help' to see available commands")
        raise typer.Exit(1)


def show_examples() -> None:
    """Display usage examples."""
    examples = """
CODESEARCH EXAMPLES

1. SEMANTIC SEARCH
   Find code using natural language:
   $ codesearch search "function that validates email addresses"
   $ codesearch search "error handling" --limit 20

2. FIND SIMILAR CODE
   Locate similar implementations:
   $ codesearch find-similar authenticate_user
   $ codesearch find-similar DatabaseConnection --language python

3. DEPENDENCY ANALYSIS
   Understand code relationships:
   $ codesearch deps src/main.py --recursive
   $ codesearch deps --format json

4. REPOSITORY INDEXING
   Prepare a repository for searching:
   $ codesearch index /home/user/projects/myapp
   $ codesearch index . --force  # Re-index current directory

5. DUPLICATE DETECTION
   Find and refactor duplicate patterns:
   $ codesearch refactor-dupes src/
   $ codesearch refactor-dupes --min-complexity 3

6. OUTPUT FORMATS
   $ codesearch search "auth" --output table  # Human readable
   $ codesearch search "auth" --output json   # Machine readable

7. CONFIGURATION
   Create custom configuration:
   $ cat > ~/.codesearch/config.json << EOF
   {
     "db_path": "/var/codesearch/db",
     "output": "json",
     "limit": 20
   }
   EOF

8. ENVIRONMENT VARIABLES
   $ export CODESEARCH_DB=/custom/db/path
   $ export CODESEARCH_OUTPUT=json
   $ codesearch search "your query"
"""
    typer.echo(examples)


def show_configuration_guide() -> None:
    """Display configuration guide."""
    guide = """
CONFIGURATION GUIDE

Codesearch can be configured through multiple methods:

1. COMMAND-LINE OPTIONS
   Most commands accept options. Example:
   $ codesearch search "query" --limit 5 --output json

2. ENVIRONMENT VARIABLES
   Set environment variables for defaults:

   CODESEARCH_DB
     Path to LanceDB database
     Default: ~/.codesearch/db

   CODESEARCH_LANGUAGE
     Default programming language filter
     Default: None (no filter)

   CODESEARCH_OUTPUT
     Default output format (table or json)
     Default: table

   CODESEARCH_LIMIT
     Default number of results
     Default: 10

   CODESEARCH_VERBOSE
     Enable verbose output (true or false)
     Default: false

3. CONFIGURATION FILES
   Codesearch looks for config files in this order:

   1. ~/.codesearch/config.yaml
   2. ~/.codesearch/config.json
   3. .codesearch/config.yaml (current directory)
   4. .codesearch/config.json (current directory)
   5. codesearch.yaml (current directory)
   6. codesearch.json (current directory)

   JSON Example:
   {
     "db_path": "/var/lib/codesearch/db",
     "output": "json",
     "limit": 20,
     "language": "python",
     "verbose": true
   }

   YAML Example:
   db_path: /var/lib/codesearch/db
   output: json
   limit: 20
   language: python
   verbose: true

4. PRIORITY
   Options are applied in this order (highest to lowest):
   1. Command-line arguments
   2. Environment variables
   3. Configuration files
   4. Default values

EXAMPLE SETUP:

# 1. Create configuration directory
mkdir -p ~/.codesearch

# 2. Create config file
cat > ~/.codesearch/config.json << EOF
{
  "db_path": "~/.codesearch/db",
  "output": "table",
  "limit": 10,
  "language": "python"
}
EOF

# 3. Index your projects
codesearch index ~/projects/myapp

# 4. Start searching
codesearch search "authentication handler"
"""
    typer.echo(guide)
