# Contributing to Codesearch

Thank you for your interest in contributing to Codesearch! This guide will help you get started with development.

## Code of Conduct

- Be respectful to all contributors
- Provide constructive feedback
- Focus on code, not the person
- Help others learn and grow

## Getting Started

### Fork & Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/your-username/codesearch.git
cd codesearch
```

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Verify installation
pytest --version
black --version
mypy --version
```

## Development Workflow

### 1. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/issue-123-description
# or for bug fixes:
git checkout -b fix/issue-456-description
```

### 2. Make Changes

Follow the guidelines below while making changes to the codebase.

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/cli/test_commands.py

# Run with coverage
pytest --cov=codesearch --cov-report=html

# Run type checks
mypy codesearch/

# Format code
black codesearch/ tests/
isort codesearch/ tests/
```

### 4. Commit Changes

```bash
# Stage changes
git add codesearch/ tests/

# Commit with descriptive message
git commit -m "Feature: Add search result filtering by complexity"

# Commit message format:
# [Type]: [Description]
# - Type: Feature, Fix, Refactor, Docs, Test, Chore
# - Description: Clear, concise (50 chars or less)
# - Additional details if needed (max 72 chars per line)
```

### 5. Push & Create PR

```bash
# Push to your fork
git push origin feature/issue-123-description

# Create pull request on GitHub
# - Title: Concise description
# - Description: Explain what and why
# - Reference related issues: "Fixes #123"
```

## Code Guidelines

### Python Style

Follow PEP 8 with these tools:

```bash
# Format code
black codesearch/ tests/

# Sort imports
isort codesearch/ tests/

# Type checking
mypy codesearch/
```

### Project Structure

```
codesearch/
├── cli/                    # CLI interface
│   ├── main.py            # Typer app entry
│   ├── commands.py        # Command implementations
│   ├── config.py          # Configuration
│   └── formatting.py      # Output formatting
├── query/                 # Query infrastructure
├── indexing/              # Data ingestion
├── embeddings/            # Embedding generation
├── parsers/               # Code parsing
├── lancedb/               # Database
├── caching/               # Caching system
├── models.py              # Data models
└── __init__.py

tests/
├── cli/                   # CLI tests
├── query/                 # Query tests
├── indexing/              # Indexing tests
└── __init__.py
```

### Naming Conventions

- **Functions**: `snake_case` (e.g., `search_pattern`)
- **Classes**: `PascalCase` (e.g., `QueryEngine`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RESULTS`)
- **Private**: Prefix with `_` (e.g., `_internal_method`)
- **Tests**: `test_<subject>.py` (e.g., `test_query_engine.py`)

### Type Hints

Always use type hints:

```python
from typing import List, Optional

def search_pattern(
    query: str,
    language: Optional[str] = None,
    limit: int = 10
) -> List[SearchResult]:
    """Search for code matching a pattern.

    Args:
        query: Natural language search query
        language: Programming language filter (optional)
        limit: Maximum results to return

    Returns:
        List of matching code entities

    Raises:
        QueryError: If query is invalid
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """Short description of what function does.

    Longer description if needed. Explain the purpose, behavior,
    and any important details about the function.

    Args:
        param1: Description of param1
        param2: Description of param2 (default: 0)

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer

    Example:
        >>> my_function("test", 5)
        True
    """
    ...
```

## Testing

### Writing Tests

Always write tests for new code:

```python
import pytest
from codesearch.query.engine import QueryEngine

class TestQueryEngine:
    """Test suite for QueryEngine."""

    @pytest.fixture
    def engine(self):
        """Create test engine."""
        return QueryEngine(db_path=":memory:")

    def test_search_pattern_basic(self, engine):
        """Test basic pattern search."""
        results = engine.search_pattern(
            query="validation",
            limit=10
        )
        assert len(results) >= 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_pattern_empty_query(self, engine):
        """Test that empty query raises error."""
        with pytest.raises(QueryError):
            engine.search_pattern(query="")

    def test_search_pattern_with_filter(self, engine):
        """Test pattern search with language filter."""
        results = engine.search_pattern(
            query="validation",
            filters=[LanguageFilter(languages=["python"])]
        )
        assert all(r.language == "python" for r in results)
```

### Test Coverage

Aim for >90% coverage:

```bash
# Generate coverage report
pytest --cov=codesearch --cov-report=html

# View report
open htmlcov/index.html
```

### Running Tests

```bash
# All tests
pytest

# Specific test
pytest tests/cli/test_commands.py::test_pattern_command

# With output
pytest -v

# Stop on first failure
pytest -x

# Run in parallel
pytest -n auto
```

## Documentation

### Update Relevant Docs

When making changes, update documentation:

- **New command**: Update [docs/CLI.md](docs/CLI.md)
- **New API**: Update [docs/API.md](docs/API.md)
- **Architecture change**: Update [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **New feature**: Add to [README.md](README.md)

### Documentation Format

- Use Markdown for all docs
- Include code examples
- Keep lines under 80 characters
- Use clear, simple language

## Commit Guidelines

### Commit Message Format

```
[TYPE]: Brief description

Longer description explaining what was changed and why.
Can span multiple lines and use bullet points:
- First change
- Second change

Fixes #123
```

### Types

- **Feature**: New functionality
- **Fix**: Bug fix
- **Refactor**: Code restructuring without behavior change
- **Docs**: Documentation updates
- **Test**: Test additions or modifications
- **Chore**: Dependencies, build system, etc.

### Examples

```
Feature: Add metadata filtering to search results

Implement MetadataFilter class for filtering code entities
by language, file path, and repository. Supports OR logic
within filter types and AND logic between types.

Fixes #45

Fix: Handle empty query strings gracefully

Validate query strings in QueryEngine.search_pattern()
to raise QueryError instead of returning empty results.

Adds test_search_pattern_empty_query test case.

Refactor: Extract database connection to separate class

Move database connection logic from QueryEngine to
DatabaseConnection class for better separation of concerns.

No behavior change, existing tests pass.
```

## Pull Request Guidelines

### Before Creating PR

- [ ] All tests pass: `pytest`
- [ ] Code formatted: `black . && isort .`
- [ ] Type checking passes: `mypy codesearch/`
- [ ] Branch is up to date: `git pull origin main`
- [ ] Docs updated if necessary
- [ ] Changelog updated if applicable

### PR Title & Description

**Good PR title:**
```
Feature: Add semantic search by query description
```

**Good PR description:**
```
## Summary
Implements semantic search functionality allowing users to find code
by natural language description rather than exact function names.

## Changes
- Add SearchResult dataclass for representing search results
- Implement QueryEngine.search_pattern() method
- Add 5 new tests for search functionality
- Update CLI to support `pattern` command

## Testing
- All 31 existing tests pass
- 5 new tests added (100% coverage of new code)
- Manual testing: Verified with 3 different queries

## Related
Fixes #12 (Issue: Implement Query Infrastructure)
```

## Areas for Contribution

### Priority Issues

- **Issue #11**: Query Infrastructure - Implement real SearchResult and MetadataFilter
- **Issue #10**: Data Ingestion Pipeline - Implement indexing orchestration
- **Issue #9**: Query Implementation - Connect search to database
- **Issue #8**: Performance - Add caching and optimization

### Types of Contributions

1. **Code Features**: Implement planned components
2. **Bug Fixes**: Fix reported issues
3. **Tests**: Improve test coverage
4. **Documentation**: Update or clarify docs
5. **Performance**: Optimize slow operations
6. **Accessibility**: Improve CLI usability

## Code Review Process

1. You submit a PR
2. Maintainers review your changes
3. You address feedback (if any)
4. PR is approved and merged
5. Your contribution is live!

### What Reviewers Look For

- ✅ Code follows guidelines
- ✅ Tests are comprehensive
- ✅ Documentation is clear
- ✅ No breaking changes
- ✅ Performance implications considered
- ✅ Commits are clean and logical

## Questions?

- Check [README.md](README.md) for project overview
- Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
- Review existing code for patterns
- Open a GitHub discussion

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct Summary

Be respectful, constructive, and inclusive. No harassment, discrimination, or bad faith arguments.

---

Thank you for contributing to Codesearch! 🎉

**Happy coding!**
