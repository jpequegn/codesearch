# Troubleshooting Guide - Codesearch

Common issues and solutions for Codesearch.

## Installation & Setup

### Issue: `pip install` fails with missing dependencies

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement transformers>=4.30.0
```

**Solution:**
1. Update pip: `pip install --upgrade pip`
2. Install dependencies separately: `pip install torch transformers`
3. Then install codesearch: `pip install -e .`

**Prevention:**
- Use Python 3.8 or later
- Create a fresh virtual environment
- Check internet connection for package downloads

---

### Issue: "No module named 'codesearch'"

**Symptoms:**
```
ModuleNotFoundError: No module named 'codesearch'
```

**Solution:**
1. Ensure installation is complete: `pip install -e .`
2. Check you're in the correct virtual environment
3. Verify the codesearch directory exists: `ls codesearch/`
4. Try reinstalling: `pip uninstall codesearch && pip install -e .`

**Prevention:**
- Always activate virtual environment before running
- Run install from repository root directory

---

### Issue: torch/transformers installation fails on macOS ARM64

**Symptoms:**
```
ERROR: Failed building wheel for torch
```

**Solution:**
1. Use conda instead: `conda install torch transformers`
2. Or use pre-built wheels: `pip install --only-binary :all: torch`

**Prevention:**
- Use conda environment for M1/M2/M3 Macs
- Update to latest pip/setuptools: `pip install --upgrade pip setuptools wheel`

---

## Database Issues

### Issue: "Database not found" error

**Symptoms:**
```
DatabaseError: Database not found at ~/.codesearch
```

**Solution:**
1. Initialize database by indexing a repository:
   ```bash
   codesearch index /path/to/repo
   ```

2. Or manually create the database:
   ```bash
   mkdir -p ~/.codesearch
   ```

**Prevention:**
- Always run `codesearch index` before searching
- Keep database in a persistent location

---

### Issue: Database permission denied

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: '~/.codesearch'
```

**Solution:**
```bash
# Fix permissions
chmod 755 ~/.codesearch
chmod 644 ~/.codesearch/*

# Or use a different location
export CODESEARCH_DB_PATH=/tmp/codesearch
```

**Prevention:**
- Use consistent database location
- Don't use system protected directories (e.g., /root/codesearch on non-root)

---

### Issue: Database corruption

**Symptoms:**
```
DatabaseError: Schema mismatch or corrupted tables
```

**Solution:**
1. Backup current database:
   ```bash
   cp -r ~/.codesearch ~/.codesearch.backup
   ```

2. Clear and rebuild:
   ```bash
   rm -rf ~/.codesearch
   codesearch index /path/to/repo
   ```

**Prevention:**
- Don't delete files while indexing
- Don't interrupt indexing with Ctrl+C
- Use `--full-reindex` flag when in doubt

---

### Issue: Database size grows too large

**Symptoms:**
- Database > 100GB
- Slow queries
- Low disk space

**Solution:**
1. Identify largest repositories:
   ```bash
   codesearch statistics --by-repo
   ```

2. Remove unused repositories:
   ```bash
   codesearch delete-repo repo-id
   ```

3. Or clear specific repositories:
   ```bash
   rm -rf ~/.codesearch
   codesearch index /path/to/kept-repo
   ```

**Prevention:**
- Remove test repositories after testing
- Monitor database size: `du -sh ~/.codesearch`
- Use separate databases for large projects

---

## Search Issues

### Issue: No search results

**Symptoms:**
```
No results found for query "validation"
```

**Possible Causes & Solutions:**

**1. Empty database**
```bash
# Check if database has any entities
codesearch statistics

# If empty, index a repository
codesearch index /path/to/repo
```

**2. Wrong language filter**
```bash
# Check what languages are indexed
codesearch statistics

# Search without language filter
codesearch pattern "validation"  # No --language flag

# Or try different language
codesearch pattern "validation" --language typescript
```

**3. Query too specific**
```bash
# Try broader query
codesearch pattern "validation"         # Instead of "email validation regex"

# Or use find-similar for known entities
codesearch find-similar validate_input
```

**4. Threshold too high**
```bash
# Lower similarity threshold
codesearch pattern "validation" --threshold 0.5
```

---

### Issue: Wrong search results

**Symptoms:**
- Results don't match the query
- Low quality matches

**Solution:**
1. Lower the threshold:
   ```bash
   codesearch pattern "email" --threshold 0.5
   ```

2. Use more specific descriptions:
   ```bash
   # Better
   codesearch pattern "function that validates email addresses with regex"

   # Worse
   codesearch pattern "email"
   ```

3. Filter by language:
   ```bash
   codesearch pattern "validation" --language python
   ```

4. Check with find-similar:
   ```bash
   codesearch find-similar known_function
   ```

---

### Issue: Search is slow

**Symptoms:**
- Queries take >5 seconds
- Commands seem to hang

**Solutions:**

**1. Check database size**
```bash
du -sh ~/.codesearch
ls -lh ~/.codesearch/*.db
```

**2. Verify database path**
```bash
echo $CODESEARCH_DB_PATH
# Should point to fast storage (SSD, not network drive)
```

**3. Move to faster storage**
```bash
# Move database to SSD
mv ~/.codesearch /mnt/ssd/.codesearch
export CODESEARCH_DB_PATH=/mnt/ssd/.codesearch
```

**4. Reduce result size**
```bash
# Limit results (fewer = faster)
codesearch pattern "validation" --limit 5
```

**5. Add language filter**
```bash
# Filtering reduces search space
codesearch pattern "validation" --language python
```

---

## Indexing Issues

### Issue: Indexing hangs or stops

**Symptoms:**
- `codesearch index` doesn't complete
- Process uses 100% CPU but makes no progress

**Solution:**
1. Stop the process: `Ctrl+C`
2. Check for errors: Look at console output
3. Restart with verbose logging:
   ```bash
   codesearch index /path/to/repo --verbose
   ```

**Prevention:**
- Index smaller repositories first
- Check available disk space: `df -h`
- Avoid indexing while other heavy processes run

---

### Issue: Indexing fails on specific files

**Symptoms:**
```
Failed to index /path/to/file.py: Parse error
```

**Solutions:**

**1. Skip problematic file**
```bash
codesearch index /path/to/repo --exclude file.py
```

**2. Skip file pattern**
```bash
codesearch index /path/to/repo --exclude "*.test.py"
```

**3. Check file syntax**
```bash
python -m py_compile file.py  # For Python files
```

**4. Increase error tolerance**
```bash
# Continue indexing even with errors
codesearch index /path/to/repo --skip-errors
```

---

### Issue: Memory usage too high during indexing

**Symptoms:**
- System becomes unresponsive
- Out of memory errors
- Indexing very slow

**Solution:**
1. Reduce batch size:
   ```bash
   codesearch index /path/to/repo --batch-size 8
   ```

2. Reduce workers:
   ```bash
   codesearch index /path/to/repo --workers 2
   ```

3. Index in smaller chunks:
   ```bash
   # Index subdirectories separately
   codesearch index /path/to/repo/src --repo-id repo-src
   codesearch index /path/to/repo/lib --repo-id repo-lib
   ```

---

### Issue: Indexing never finishes for large repository

**Symptoms:**
- Still indexing after hours
- Progress seems stuck

**Solution:**
1. Cancel and try with fewer workers:
   ```bash
   Ctrl+C
   codesearch index /path/to/repo --workers 2 --batch-size 16
   ```

2. Try incremental only:
   ```bash
   codesearch index /path/to/repo --incremental --skip-errors
   ```

3. Split into multiple indexes:
   ```bash
   codesearch index /path/a --repo-id repo-a
   codesearch index /path/b --repo-id repo-b
   ```

---

## Configuration Issues

### Issue: Environment variables not working

**Symptoms:**
```bash
export CODESEARCH_DB_PATH=/custom/path
codesearch pattern "test"  # Still uses ~/.codesearch
```

**Solution:**
1. Verify export worked:
   ```bash
   echo $CODESEARCH_DB_PATH
   ```

2. Set in shell profile for persistence:
   ```bash
   echo 'export CODESEARCH_DB_PATH=/custom/path' >> ~/.bashrc
   source ~/.bashrc
   ```

3. Pass as command line argument:
   ```bash
   codesearch index /path/to/repo --database /custom/path
   ```

---

### Issue: Wrong output format

**Symptoms:**
```bash
codesearch pattern "test" --format json  # But output is table
```

**Solution:**
1. Check if format is supported:
   ```bash
   codesearch pattern --help | grep format
   ```

2. Use capitalization correctly:
   ```bash
   codesearch pattern "test" --format json  # json (lowercase)
   codesearch pattern "test" --format table # table (lowercase)
   ```

3. Set default format:
   ```bash
   export CODESEARCH_OUTPUT_FORMAT=json
   ```

---

## Performance Issues

### Issue: Commands are very slow

**Symptoms:**
- All commands take >1 second
- Response delays are noticeable

**Solutions (in order of impact):**

1. **Move database to SSD**
   ```bash
   mv ~/.codesearch /ssd/path/.codesearch
   export CODESEARCH_DB_PATH=/ssd/path/.codesearch
   ```

2. **Add language filter**
   ```bash
   codesearch pattern "validation" --language python
   ```

3. **Reduce database size**
   ```bash
   codesearch delete-repo unused-repo
   ```

4. **Rebuild database index**
   ```bash
   codesearch rebuild-index
   ```

---

## Error Messages

### "DatabaseError: Connection failed"

**Cause:** Database not accessible
**Solution:**
- Check database path exists
- Check read/write permissions
- Verify disk space available

### "QueryError: Invalid query"

**Cause:** Malformed search parameters
**Solution:**
- Check query syntax
- Use quotes for multi-word queries: `"email validation"`
- Verify --language is valid (python, typescript, go)

### "ConfigurationError: Invalid argument"

**Cause:** Invalid command-line arguments
**Solution:**
- Check command syntax: `codesearch <command> --help`
- Verify argument values are valid
- Check for typos in flag names

### "EmbeddingError: Model not found"

**Cause:** Embedding model couldn't be downloaded
**Solution:**
- Check internet connection
- Try again (model download sometimes fails)
- Specify alternative model: `--model sentence-transformers/all-MiniLM-L6-v2`

---

## Getting Help

### Debugging Steps

1. **Enable verbose mode:**
   ```bash
   codesearch --verbose pattern "test"
   ```

2. **Check logs:**
   ```bash
   tail -f ~/.codesearch/logs/codesearch.log
   ```

3. **Collect diagnostic info:**
   ```bash
   codesearch diagnose
   ```

4. **Test with simple query:**
   ```bash
   codesearch pattern "function"
   ```

### Reporting Issues

When reporting issues, include:
1. Codesearch version: `codesearch --version`
2. Python version: `python --version`
3. OS: `uname -a`
4. Full error message
5. Steps to reproduce
6. Output of `codesearch diagnose`

### Resources

- [README](../README.md) - Project overview
- [Architecture](ARCHITECTURE.md) - System design
- [CLI Reference](CLI.md) - Command documentation
- [Python API](API.md) - Programmatic usage
- GitHub Issues: https://github.com/your-username/codesearch/issues

---

## Tips & Tricks

### Optimize search performance

```bash
# Use specific language to speed up search
codesearch pattern "validation" --language python

# Limit results early
codesearch pattern "error" --limit 5

# Use higher threshold for faster searches
codesearch pattern "handler" --threshold 0.85
```

### Manage multiple codebases

```bash
# Create separate databases
CODESEARCH_DB_PATH=~/.codesearch-repo1 codesearch index ~/repo1
CODESEARCH_DB_PATH=~/.codesearch-repo2 codesearch index ~/repo2

# Query specific database
CODESEARCH_DB_PATH=~/.codesearch-repo1 codesearch pattern "validation"
```

### Batch processing

```bash
# Create script to search multiple terms
#!/bin/bash
for query in "validation" "error handling" "database query"; do
  echo "=== Searching for: $query ==="
  codesearch pattern "$query" --limit 3
  echo ""
done
```

### Monitor database growth

```bash
# Check size
du -sh ~/.codesearch

# Get entity count
codesearch statistics
```

---

## FAQ

**Q: Can I use Codesearch without internet?**
A: Yes, models are downloaded once and stored locally. After first run, no internet needed.

**Q: How much disk space does the database use?**
A: ~10-20 MB per 1,000 entities. Large projects (50K+ entities) may use 500MB-2GB.

**Q: Can I search across multiple repositories?**
A: Yes, index all repos to the same database with different `--repo-id` values.

**Q: Is the source code exposed in results?**
A: No, only metadata and embeddings are stored. Actual code is referenced by file/line only.

**Q: Can I delete or update indexed code?**
A: Yes, re-index with `--full-reindex` to update all code.

**Q: What programming languages are supported?**
A: Python, TypeScript, Go (extensible to others).

---

## Summary

| Issue | Quick Fix | Prevention |
|-------|-----------|-----------|
| No database | `codesearch index /path` | Index before searching |
| No results | Check database with `codesearch statistics` | Ensure code is indexed |
| Slow searches | Add language filter | Filter early in queries |
| High memory | Use `--batch-size 8 --workers 2` | Avoid huge batch operations |
| Indexing hangs | Ctrl+C, restart with `--workers 2` | Monitor long-running operations |

---

**Still having issues?** Check the troubleshooting checklist:

- ✓ Virtual environment activated
- ✓ Codesearch installed: `pip install -e .`
- ✓ Database initialized: `codesearch index /path`
- ✓ Correct database path: `echo $CODESEARCH_DB_PATH`
- ✓ Sufficient disk space: `df -h`
- ✓ Read/write permissions: `ls -l ~/.codesearch`
