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

```bash
# Installation
pip install -e .

# Index a repository
codesearch index /path/to/repo

# Find similar functions
codesearch find-similar my_function_name

# Semantic search by description
codesearch pattern "function that validates email addresses"

# Show call dependencies
codesearch dependencies my_function_name

# Find duplicate/similar code
codesearch refactor-dupes
```

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

See [docs/architecture.md](docs/architecture.md) for detailed design documentation.

**Core Components:**
1. **Code Parser & Indexer** - Extracts functions, classes, and call relationships
2. **Embedding Pipeline** - Generates semantic embeddings for code
3. **LanceDB Database** - Stores vectors and metadata
4. **CLI Query Interface** - User-facing command-line tool
5. **Indexing System** - Incremental updates and multi-repo support

## Development

```bash
# Setup development environment
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black . && isort .

# Type checking
mypy codesearch/
```

## Project Status

🚧 **Under Development** - Currently in active development. See [GitHub Issues](https://github.com/your-username/codesearch/issues) for current work.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
