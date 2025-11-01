# Codesearch Implementation Guide

## Overview

The codesearch tool processes Python code through 4 main stages:

```
Python Codebase
    ↓
[1] Parse with AST → Extract functions + metadata
    ↓
[2] Generate embeddings → Semantic representations
    ↓
[3] Store in LanceDB → Indexed and queryable
    ↓
[4] Query via CLI → Find similar code, patterns, dependencies
```

## Stage 1: Parsing Python Code

**Goal**: Extract all functions with full context

**Input**: Python file path
**Output**: List of Function objects with metadata

```python
# codesearch/models.py
@dataclass
class Function:
    name: str
    file_path: str
    language: str = "python"

    # Code content
    source_code: str
    docstring: Optional[str]

    # Metadata
    line_number: int
    end_line: int
    signature: str  # e.g., "def foo(a: int, b: str) -> bool:"

    # Call graph
    calls_to: List[str]  # Functions this calls
    called_by: List[str]  # Functions that call this

    # Embedding
    embedding: Optional[List[float]] = None
```

**Parser Implementation**:

```python
# codesearch/parsers/python_parser.py
import ast
from typing import List

class PythonParser:
    def parse_file(self, file_path: str) -> List[Function]:
        """Extract all functions from a Python file."""
        with open(file_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func = self._extract_function(node, source, file_path)
                functions.append(func)

        return functions

    def _extract_function(self, node: ast.FunctionDef, source: str, file_path: str) -> Function:
        """Extract metadata for a single function."""
        source_lines = source.split('\n')
        source_code = '\n'.join(source_lines[node.lineno-1:node.end_lineno])

        return Function(
            name=node.name,
            file_path=file_path,
            source_code=source_code,
            docstring=ast.get_docstring(node),
            line_number=node.lineno,
            end_line=node.end_lineno,
            signature=self._get_signature(node),
            calls_to=self._get_calls(node),
            called_by=[]  # Populated in call graph phase
        )

    def _get_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature."""
        args = [arg.arg for arg in node.args.args]
        return f"def {node.name}({', '.join(args)}):"

    def _get_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extract functions called by this function."""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
        return list(set(calls))  # Deduplicate
```

## Stage 2: Generate Embeddings

**Goal**: Convert function code to vector representations

**Input**: Function objects
**Output**: Same functions with embeddings populated

```python
# codesearch/embeddings/embedder.py
from transformers import AutoTokenizer, AutoModel
import torch

class CodeEmbedder:
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed_function(self, func: Function) -> Function:
        """Generate embedding for a function."""
        # Combine code + docstring for richer semantic meaning
        text = f"{func.source_code}\n{func.docstring or ''}"

        # Tokenize with truncation
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embedding = embedding.cpu().numpy().flatten().tolist()

        func.embedding = embedding
        return func

    def embed_batch(self, functions: List[Function], batch_size: int = 32) -> List[Function]:
        """Embed multiple functions efficiently."""
        for i in range(0, len(functions), batch_size):
            batch = functions[i:i+batch_size]
            for func in batch:
                self.embed_function(func)
        return functions
```

## Stage 3: Store in LanceDB

**Goal**: Index functions for fast semantic search

**Input**: Embedded functions
**Output**: Searchable LanceDB table

```python
# codesearch/database/lance_db.py
import lancedb
from typing import List, Dict

class LanceDBManager:
    def __init__(self, db_path: str = ".lancedb"):
        self.db = lancedb.connect(db_path)
        self.table_name = "functions"

    def create_table(self, functions: List[Function]):
        """Create and populate the functions table."""
        data = [
            {
                "id": f"{func.file_path}:{func.line_number}",
                "name": func.name,
                "file_path": func.file_path,
                "language": func.language,
                "source_code": func.source_code,
                "docstring": func.docstring,
                "line_number": func.line_number,
                "signature": func.signature,
                "calls_to": func.calls_to,
                "called_by": func.called_by,
                "embedding": func.embedding,
            }
            for func in functions
        ]

        self.table = self.db.create_table(
            self.table_name,
            data=data,
            mode="overwrite"
        )

    def search_similar(self, embedding: List[float], limit: int = 10) -> List[Dict]:
        """Find functions semantically similar to embedding."""
        results = self.table.search(embedding).limit(limit).to_list()
        return results

    def search_by_metadata(self, filters: Dict) -> List[Dict]:
        """Filter functions by metadata (language, file, etc)."""
        query = self.table.search()

        if "language" in filters:
            query = query.where(f"language = '{filters['language']}'")
        if "file_path" in filters:
            query = query.where(f"file_path LIKE '%{filters['file_path']}%'")

        return query.to_list()

    def get_function(self, name: str) -> Dict:
        """Retrieve a specific function by name."""
        results = self.table.search().where(f"name = '{name}'").to_list()
        return results[0] if results else None
```

## Stage 4: CLI Interface

**Goal**: User-friendly query interface

```python
# codesearch/cli/main.py
import click
from codesearch.parsers.python_parser import PythonParser
from codesearch.embeddings.embedder import CodeEmbedder
from codesearch.database.lance_db import LanceDBManager
from codesearch.indexing.indexer import Indexer

@click.group()
def main():
    """Codesearch - Semantic code search for Python."""
    pass

@main.command()
@click.argument('repo_path')
def index(repo_path):
    """Index a Python repository."""
    click.echo(f"Indexing {repo_path}...")

    # Stage 1: Parse
    indexer = Indexer()
    functions = indexer.parse_directory(repo_path)
    click.echo(f"Extracted {len(functions)} functions")

    # Stage 2: Embed
    embedder = CodeEmbedder()
    functions = embedder.embed_batch(functions)
    click.echo(f"Generated embeddings")

    # Stage 3: Store
    db = LanceDBManager()
    db.create_table(functions)
    click.echo(f"Stored in LanceDB")

@main.command()
@click.argument('function_name')
def find_similar(function_name):
    """Find functions similar to the given function."""
    db = LanceDBManager()

    # Get the function
    func = db.get_function(function_name)
    if not func:
        click.echo(f"Function {function_name} not found")
        return

    # Search for similar
    results = db.search_similar(func['embedding'], limit=10)

    click.echo(f"\nFunctions similar to {function_name}:")
    for result in results:
        click.echo(f"  - {result['name']} ({result['file_path']})")

@main.command()
@click.argument('description')
def pattern(description):
    """Search for functions matching a semantic description."""
    embedder = CodeEmbedder()
    db = LanceDBManager()

    # Embed the description
    inputs = embedder.tokenizer(
        description,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = embedder.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten().tolist()

    # Search
    results = db.search_similar(embedding, limit=10)

    click.echo(f"\nFunctions matching '{description}':")
    for result in results:
        click.echo(f"  - {result['name']} ({result['file_path']})")
        if result['docstring']:
            click.echo(f"    {result['docstring'][:60]}...")

if __name__ == '__main__':
    main()
```

## Implementation Workflow

### 1. Start with Python Parser (Issue #1)
```bash
codesearch/parsers/
├── __init__.py
├── base.py           # Abstract parser interface
└── python_parser.py  # Python-specific implementation
```

### 2. Add Models & Database (Issues #2-3, #9)
```bash
codesearch/
├── models.py         # Function dataclass
├── embeddings/
│   ├── __init__.py
│   └── embedder.py
└── database/
    ├── __init__.py
    └── lance_db.py
```

### 3. Build Indexer (Issues #1-4, #17)
```bash
codesearch/indexing/
├── __init__.py
├── scanner.py        # Walk directory tree
├── indexer.py        # Orchestrate parsing + embedding + storage
└── call_graph.py     # Build dependency relationships
```

### 4. CLI Interface (Issues #13-16)
```bash
codesearch/cli/
├── __init__.py
└── main.py           # Click commands
```

## Quick Test Workflow

Once basic structure is in place:

```bash
# 1. Index a real Python project
codesearch index ~/Code/parakeet-podcast-processor

# 2. Find similar functions
codesearch find-similar validate_episodes

# 3. Search by description
codesearch pattern "function that checks if file exists and is readable"

# 4. Show dependencies
codesearch dependencies fetch_episodes
```

## Key Implementation Details

**Python AST parsing**:
- Use `ast.walk()` to traverse the tree
- Extract docstrings with `ast.get_docstring()`
- Get source code from line numbers
- Track calls via `ast.Call` nodes

**Embedding**:
- CodeBERT (768 dimensions) - best for code semantics
- Truncate long functions to 512 tokens
- Cache embeddings to avoid recomputing

**LanceDB storage**:
- Vector field for embeddings (auto-indexed)
- Metadata fields for filtering
- Simple schema: id, name, file_path, source_code, embedding

**CLI design**:
- Each command is self-contained
- Load database once per command
- Pretty output with click formatting
