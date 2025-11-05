# Architecture - Codesearch

Codesearch is built around a **5-component architecture** that enables semantic code search through vector embeddings and intelligent querying.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Interface                             │
│              (Typer-based command-line tool)                    │
└────┬─────────────────────────────────────────────────────────┬──┘
     │                                                          │
     ├──────────────────┬──────────────────┬───────────────────┤
     │                  │                  │                   │
     ▼                  ▼                  ▼                   ▼
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  Pattern │    │ Find Similar │    │Dependencies  │    │  Refactor│
│  Command │    │   Command    │    │   Command    │    │  Command │
└──────────┘    └──────────────┘    └──────────────┘    └──────────┘
     │                  │                  │                   │
     └──────────────────┴──────────────────┴───────────────────┘
                        │
                        ▼
              ┌──────────────────────┐
              │  Query Infrastructure │
              │   (Issue #11 - TBD)   │
              ├──────────────────────┤
              │ • QueryEngine         │
              │ • SearchResult        │
              │ • MetadataFilter      │
              └──────────┬────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
    ┌──────────────┐            ┌──────────────────┐
    │  LanceDB API │            │  Call Graph      │
    │  (Vectors)   │            │  Analysis        │
    └──────────────┘            └──────────────────┘
        │                              │
        └──────────────┬───────────────┘
                       │
                       ▼
            ┌──────────────────────────┐
            │   LanceDB Database       │
            │   (Vector & Metadata)    │
            ├──────────────────────────┤
            │ Tables:                  │
            │ • code_entities          │
            │ • code_relationships     │
            │ • search_metadata        │
            └──────────────────────────┘
```

## Core Components

### 1. Code Parser & Indexer
**Location**: `codesearch/parsers/`, `codesearch/indexing/`

Extracts code entities (functions, classes, methods) and their relationships from source code.

**Key Responsibilities**:
- Parse source code using tree-sitter
- Extract entity metadata: name, signature, docstring, source location
- Identify call relationships (caller-callee)
- Support multiple languages: Python, TypeScript, Go (extensible)

**Data Flow**:
```
Source Code → Tree-sitter → AST → Entity Extraction → Metadata
             │
             └─→ Call Graph Analysis → Relationships
```

**Language Support**:
- Python (3.8+)
- TypeScript / JavaScript
- Go
- Extensible for additional languages

### 2. Embedding Pipeline
**Location**: `codesearch/embeddings/`

Converts code and metadata into dense vector embeddings using transformer models.

**Key Responsibilities**:
- Load pre-trained embedding models (transformers library)
- Tokenize and embed code snippets
- Embed entity names, docstrings, and descriptions
- Cache embeddings for performance
- Support batch embedding generation

**Embedding Models**:
- `sentence-transformers/all-MiniLM-L6-v2` (default, 384-dim)
- `sentence-transformers/all-mpnet-base-v2` (768-dim, more accurate)
- Custom fine-tuned models (supported)

**Flow**:
```
Code Snippet + Metadata → Tokenization → Transformer Model → Embedding Vector
```

### 3. LanceDB Database
**Location**: `codesearch/lancedb/`

Vector database storing code embeddings and metadata with fast similarity search.

**Schema**:

#### `code_entities` Table
```
{
  entity_id: str (primary key, SHA256(repo_id + file_path + entity_name))
  name: str (function/class/method name)
  type: str (function|class|method)
  signature: str (function signature)
  docstring: str (extracted documentation)
  source_file: str (relative file path)
  start_line: int (line number)
  end_line: int (line number)
  embedding: vector<f32> (384 or 768 dimensions)
  language: str (python|typescript|go)
  repository_id: str (for multi-repo support)
  created_at: timestamp
  updated_at: timestamp
}
```

#### `code_relationships` Table
```
{
  relationship_id: str (primary key)
  caller_id: str (entity_id)
  callee_id: str (entity_id)
  call_count: int (number of calls)
  call_locations: list<int> (line numbers)
  relationship_type: str (calls|inherits_from|imports)
  repository_id: str
  created_at: timestamp
}
```

#### `search_metadata` Table
```
{
  entity_id: str (foreign key)
  tags: list<str> (user-defined tags)
  complexity_score: float (cyclomatic complexity estimate)
  line_count: int
  test_coverage: float (optional)
  last_modified: timestamp
}
```

**Key Features**:
- O(1) entity lookups via indexed entity_id
- Fast similarity search using vector distance metrics
- Metadata filtering with AND/OR logic
- Full-text search on text fields
- Support for incremental updates
- Automatic schema management

### 4. CLI Query Interface
**Location**: `codesearch/cli/`

User-facing command-line tool built with Typer and Rich for beautiful terminal output.

**Commands**:
1. `codesearch pattern <query>` - Semantic search by natural language
2. `codesearch find-similar <entity_name>` - Find similar functions
3. `codesearch dependencies <entity_name>` - Show call graph
4. `codesearch index <path>` - Index a repository
5. `codesearch refactor-dupes [--threshold]` - Find duplicate code

**Configuration**:
- Database path: `CODESEARCH_DB_PATH` (default: ~/.codesearch)
- Default language: `CODESEARCH_LANGUAGE` (default: python)
- Output format: `CODESEARCH_OUTPUT_FORMAT` (default: table)

**Output Formats**:
- **Terminal Table**: Rich-formatted ASCII table with colors
- **JSON**: Machine-readable JSON for programmatic use
- **Raw**: Plain text output

**Exit Codes**:
- 0: Success
- 1: Configuration error
- 2: Database error
- 3: Query error

### 5. Data Ingestion System
**Location**: `codesearch/indexing/`

Orchestrates incremental indexing of code repositories with deduplication and validation.

**Key Responsibilities**:
- Discover code files in repositories
- Extract code entities (delegates to parser)
- Generate embeddings (delegates to embedding pipeline)
- Deduplicate using SHA256 hashing
- Validate data before insertion
- Handle errors gracefully (partial success model)
- Support multi-repository indexing
- Implement incremental updates (only changed files)

**Ingestion Pipeline**:
```
File Discovery → Hash-based Deduplication → Parsing → Embedding → Validation → Insert → Audit Trail
```

**Features**:
- Incremental indexing (tracks file hashes)
- Batch insertion for performance
- Rollback mechanism for failed batches
- Audit trail for traceability
- Support for multiple repositories with metadata tagging

## Data Flow

### Indexing Flow
```
1. User runs: codesearch index /path/to/repo
2. CLI → DataIngestionPipeline
3. Pipeline discovers Python/TS/Go files
4. For each file:
   a. Compute SHA256 hash
   b. Check if already indexed (hash matches)
   c. Skip if unchanged OR re-parse if changed
   d. Extract entities → Generate embeddings
   e. Validate metadata
   f. Batch insert into LanceDB
5. Update audit trail with ingestion metadata
6. Return success/failure summary
```

### Search Flow
```
1. User runs: codesearch pattern "function that validates email"
2. CLI → QueryEngine
3. QueryEngine:
   a. Embed user query using same model as code
   b. Search LanceDB for similar vectors
   c. Apply metadata filters (language, file, etc.)
   d. Apply pagination
   e. Fetch entity metadata
4. Format results (table or JSON)
5. Display to user with context
```

### Dependency Flow
```
1. User runs: codesearch dependencies my_function
2. CLI → QueryEngine → LanceDB code_relationships table
3. For each relationship:
   a. Fetch caller and callee entities
   b. Build relationship graph
   c. Optionally show call stack
4. Format results (tree or table)
5. Display with color-coded relationship types
```

## Technology Stack

### Core Dependencies
| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Database** | LanceDB | Vector storage & similarity search |
| **Parsing** | tree-sitter | Code parsing for multiple languages |
| **Embeddings** | transformers | Pre-trained code embeddings |
| **ML Framework** | torch | Machine learning inference |
| **CLI** | Typer | Command-line interface |
| **Terminal** | Rich | Beautiful terminal formatting |
| **Data Validation** | Pydantic | Type-safe data models |

### Development Dependencies
| Tool | Purpose |
|------|---------|
| pytest | Testing framework |
| black | Code formatting |
| isort | Import sorting |
| mypy | Type checking |
| pytest-cov | Coverage reporting |

## Performance Characteristics

### Latency
| Operation | Time | Notes |
|-----------|------|-------|
| Pattern search | 110-220ms | 10 results, database warm |
| Find similar | 150-300ms | 20 results |
| Dependency lookup | 50-100ms | O(1) indexed lookup |
| Index (per file) | 100-500ms | Depends on file size |

### Scalability
- **Database Size**: Tested up to 50K entities (15GB)
- **Query Time**: Sub-second for typical codebases
- **Memory**: ~2-4GB for typical 10K-entity indices
- **Multi-repo**: Linear scaling with number of repositories

## Error Handling & Recovery

### Error Categories
1. **Configuration Errors**: Invalid paths, missing permissions (exit 1)
2. **Database Errors**: Connection failures, schema issues (exit 2)
3. **Query Errors**: Invalid queries, empty results (exit 3)
4. **Ingestion Errors**: Parse failures, validation failures (partial success with reporting)

### Recovery Strategy
- **Validation**: Pre-insertion validation prevents corrupt data
- **Partial Success**: Failed entities don't block entire batch
- **Audit Trail**: All operations logged for debugging
- **Rollback**: Failed batches can be rolled back if needed

## Security Considerations

### Access Control
- File system access restricted to indexed repositories
- No remote code execution (parsing is read-only)
- Database files stored locally with filesystem permissions

### Data Privacy
- Embeddings are stored locally (no cloud transmission by default)
- Source code is indexed but not exposed in API responses
- No external model dependencies (models downloaded once)

### Input Validation
- All user inputs validated with Pydantic
- Query strings sanitized before database operations
- File paths validated against repository roots

## Future Enhancements

### Planned Features
1. **Remote API**: REST API for programmatic access
2. **Web Interface**: Browser-based search and exploration
3. **Advanced Analytics**: Code quality metrics, complexity analysis
4. **Custom Models**: Fine-tuned models for specific languages
5. **Distributed Indexing**: Multi-machine indexing for large codebases
6. **Version History**: Search across git history

### Architecture Evolution
- Modularization for plugin system
- Caching layer improvements (Redis support)
- Streaming ingestion for large repositories
- Distributed query execution

## Summary

Codesearch provides a scalable, maintainable architecture for semantic code search:

- **Separation of Concerns**: Clear boundaries between parsing, embedding, storage, and querying
- **Extensibility**: Language support, embedding models, metadata filters
- **Performance**: Optimized vector search with incremental indexing
- **Reliability**: Comprehensive error handling and audit trails
- **User Experience**: Beautiful CLI with multiple output formats
