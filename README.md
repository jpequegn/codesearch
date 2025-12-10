# Codesearch - Semantic Code Intelligence Tool

**Semantic code search powered by vector embeddings and LanceDB**

Find similar functions, understand code patterns, navigate dependencies, and identify refactoring opportunities using AI-powered semantic search.

## Overview

Codesearch indexes your codebase and enables intelligent queries like:

- **Find Similar Functions**: "Show me functions similar to this one" → Discover patterns and duplicates
- **Understand Patterns**: "What patterns are used for error handling?" → Learn how your code solves problems
- **Navigate Dependencies**: "What calls this function?" "What does it call?" → Trace code relationships
- **Refactor Insights**: "Find code that does the same thing differently" → Identify consolidation opportunities

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/codesearch.git
cd codesearch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Basic Usage

```bash
# 1. Index a repository
codesearch index /path/to/your/repo

# 2. Find functions similar to a specific one
codesearch find-similar my_function_name

# 3. Search semantically by description
codesearch pattern "function that validates email addresses"

# 4. Explore call dependencies
codesearch dependencies my_function_name

# 5. Find duplicate or similar code patterns
codesearch refactor-dupes --threshold 0.85
```

### Configuration

Set environment variables to customize behavior:

```bash
# Database location (default: ~/.codesearch)
export CODESEARCH_DB_PATH=~/.codesearch

# Default programming language for searches (python, typescript, go)
export CODESEARCH_LANGUAGE=python

# Output format (table for terminal, json for scripts)
export CODESEARCH_OUTPUT_FORMAT=table
```

### Example Workflows

**Discover code patterns in your project:**
```bash
# Index your codebase
codesearch index ~/my-project

# Find all error handling functions
codesearch pattern "function that handles errors or exceptions"

# Find validation functions
codesearch pattern "validates input"
```

**Find code duplicates for refactoring:**
```bash
# Index your codebase
codesearch index ~/my-project

# Find similar implementations (duplication candidates)
codesearch refactor-dupes --threshold 0.90

# Examine specific function
codesearch find-similar database_query
```

**Explore code dependencies:**
```bash
# Index your codebase
codesearch index ~/my-project

# See what calls a function
codesearch dependencies main

# Understand call graph
codesearch dependencies api_handler
```

### Command Reference

For detailed command documentation, see [docs/CLI.md](docs/CLI.md).

- **`pattern <query>`** - Search for code matching a natural language description
- **`find-similar <entity_name>`** - Find functions similar to a given function
- **`dependencies <entity_name>`** - Show functions that call a given function
- **`index <path>`** - Index a repository or directory
- **`refactor-dupes [--threshold]`** - Find potential code duplicates

## Features

- 🔍 **Semantic Search**: Find code by meaning, not just keywords
- 📊 **Call Graph Analysis**: Understand function relationships and dependencies
- 🎯 **Pattern Discovery**: Identify code patterns and anti-patterns
- 🔄 **Multi-Language Support**: Python, TypeScript, Go (extensible)
- 📈 **Incremental Indexing**: Re-index only changed files
- 💾 **Multiple Repos**: Index and search across multiple codebases
- ⚡ **Fast Queries**: Vector similarity search with LanceDB
- 🖥️ **CLI Interface**: Easy command-line access

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for comprehensive design documentation including system diagrams, data flow, and technology stack.

**Core Components:**

1. **Code Parser & Indexer** (`codesearch/parsers/`, `codesearch/indexing/`)
   - Extracts functions, classes, and call relationships from source code
   - Supports Python, TypeScript, and Go (extensible)
   - Identifies entity metadata: names, signatures, docstrings, locations

2. **Embedding Pipeline** (`codesearch/embeddings/`)
   - Generates semantic embeddings using pre-trained transformer models
   - Supports multiple models: MiniLM (384-dim), MPNet (768-dim)
   - Batch processing for performance

3. **LanceDB Database** (`codesearch/lancedb/`)
   - Vector database for semantic search
   - Tables: `code_entities`, `code_relationships`, `search_metadata`
   - Sub-second search on large codebases

4. **CLI Query Interface** (`codesearch/cli/`)
   - 5 user-facing commands: pattern, find-similar, dependencies, index, refactor-dupes
   - Multiple output formats: terminal table, JSON
   - Configuration via environment variables

5. **Data Ingestion System** (`codesearch/indexing/`)
   - Orchestrates repository indexing
   - Hash-based deduplication for efficiency
   - Incremental updates (only changed files)
   - Multi-repository support with audit trails

## Installation

For detailed installation instructions, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

**Install from source:**
```bash
git clone https://github.com/jpequegn/codesearch.git
cd codesearch
pip install -e ".[dev]"
```

## Development

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install with dev dependencies
pip install -e ".[dev]"
```

### Quality Checks

```bash
# Run all tests with coverage
pytest --cov=codesearch --cov-report=html

# Format code
black codesearch/ tests/
isort codesearch/ tests/

# Type checking
mypy codesearch/

# Lint with ruff
ruff check codesearch/ tests/
```

### Project Structure

```
codesearch/
├── cli/                    # Command-line interface
├── query/                  # Query infrastructure
├── indexing/               # Data ingestion
├── embeddings/             # Embedding generation
├── lancedb/                # Database layer
├── parsers/                # Code parsing
├── caching/                # Caching system
└── models.py               # Data models

tests/
├── cli/                    # CLI tests
├── integration/            # Integration tests
└── conftest.py             # Shared fixtures

docs/
├── ARCHITECTURE.md         # System design
├── CLI.md                  # Command reference
├── API.md                  # Python API
├── INSTALLATION.md         # Setup guide
└── TROUBLESHOOTING.md      # Troubleshooting
```

### See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
- [docs/INSTALLATION.md](docs/INSTALLATION.md) - Installation guide
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Troubleshooting

## Project Status

📊 **Component Status**:
- ✅ Component 5.1-5.5: Core features (indexing, caching, error handling, testing, documentation)
- ✅ Component 5.6: Project documentation (README, architecture, CLI, API, troubleshooting)
- ✅ Component 5.7: Project setup & deployment (this release)
- 🔄 Component #9: LanceDB schema (in progress)
- 🔄 Component #10: Data ingestion (in progress)
- 🔄 Component #11: Query infrastructure (in progress)

See [GitHub Issues](https://github.com/jpequegn/codesearch/issues) for current work.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
