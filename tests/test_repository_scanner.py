"""Tests for repository scanner."""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from codesearch.indexing.scanner import RepositoryScannerImpl
from codesearch.models import FileMetadata, RepositoryScanner


@pytest.fixture
def temp_repo():
    """Create a temporary repository with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure
        os.makedirs(os.path.join(tmpdir, "src"))
        os.makedirs(os.path.join(tmpdir, "tests"))
        os.makedirs(os.path.join(tmpdir, "__pycache__"))
        os.makedirs(os.path.join(tmpdir, "venv", "lib"))

        # Create Python files
        Path(os.path.join(tmpdir, "src", "main.py")).write_text("print('hello')\n")
        Path(os.path.join(tmpdir, "src", "utils.py")).write_text("def helper(): pass\n")
        Path(os.path.join(tmpdir, "tests", "test_main.py")).write_text("def test(): pass\n")

        # Create ignored files
        Path(os.path.join(tmpdir, "__pycache__", "main.pyc")).write_text("")
        Path(os.path.join(tmpdir, "venv", "lib", "python.py")).write_text("")
        Path(os.path.join(tmpdir, ".DS_Store")).write_text("")

        # Create non-Python files
        Path(os.path.join(tmpdir, "README.md")).write_text("# README")
        Path(os.path.join(tmpdir, "requirements.txt")).write_text("numpy==1.0\n")

        yield tmpdir


@pytest.fixture
def scanner():
    """Create a repository scanner."""
    return RepositoryScannerImpl()


@pytest.fixture
def multi_repo():
    """Create multiple temporary repositories."""
    tmpdir1 = tempfile.mkdtemp()
    tmpdir2 = tempfile.mkdtemp()

    # Repo 1: Python files
    Path(os.path.join(tmpdir1, "module1.py")).write_text("def func1(): pass\n")
    Path(os.path.join(tmpdir1, "module2.py")).write_text("def func2(): pass\n")

    # Repo 2: Python files
    Path(os.path.join(tmpdir2, "app.py")).write_text("def main(): pass\n")

    yield {"repo1": tmpdir1, "repo2": tmpdir2}

    # Cleanup
    import shutil

    shutil.rmtree(tmpdir1)
    shutil.rmtree(tmpdir2)


# ==================== File Metadata Tests ====================


def test_file_metadata_relative_path():
    """Test FileMetadata.relative_path method."""
    metadata = FileMetadata(
        file_path="/home/user/project/src/main.py",
        language="python",
        size_bytes=100,
        modified_time=datetime.now(),
    )

    rel_path = metadata.relative_path("/home/user/project")
    assert rel_path == "src/main.py"

    # Test with trailing slash
    rel_path = metadata.relative_path("/home/user/project/")
    assert rel_path == "src/main.py"


# ==================== Repository Scanner Configuration Tests ====================


def test_scanner_should_scan_directory():
    """Test directory filtering."""
    config = RepositoryScanner()

    assert config.should_scan_directory("src") is True
    assert config.should_scan_directory("tests") is True
    assert config.should_scan_directory("__pycache__") is False
    assert config.should_scan_directory(".git") is False
    assert config.should_scan_directory("venv") is False
    assert config.should_scan_directory("node_modules") is False


def test_scanner_should_scan_file():
    """Test file filtering."""
    config = RepositoryScanner()

    assert config.should_scan_file("main.py", "python") is True
    assert config.should_scan_file("test.py", "python") is True
    assert config.should_scan_file("main.pyc", "python") is False
    assert config.should_scan_file("main.js", "javascript") is False  # Unsupported language
    assert config.should_scan_file(".DS_Store", "unknown") is False


def test_scanner_get_language_from_extension():
    """Test language detection."""
    config = RepositoryScanner()

    assert config.get_language_from_extension("main.py") == "python"
    assert config.get_language_from_extension("module.pyx") == "python"
    assert config.get_language_from_extension("module.pyi") == "python"
    assert config.get_language_from_extension("main.js") == "javascript"
    assert config.get_language_from_extension("main.ts") == "typescript"
    assert config.get_language_from_extension("main.go") == "go"
    assert config.get_language_from_extension("main.rs") == "rust"
    assert config.get_language_from_extension("README.md") is None


def test_add_ignored_directory():
    """Test adding custom ignored directories."""
    config = RepositoryScanner()

    config.ignored_directories.add("custom_ignore")
    assert config.should_scan_directory("custom_ignore") is False
    assert config.should_scan_directory("src") is True


# ==================== Repository Scanning Tests ====================


def test_scan_repository(scanner, temp_repo):
    """Test basic repository scanning."""
    files = scanner.scan_repository(temp_repo)

    # Should find 3 Python files (not in ignored dirs/with ignored patterns)
    assert len(files) == 3
    file_names = {os.path.basename(f.file_path) for f in files}
    assert "main.py" in file_names
    assert "utils.py" in file_names
    assert "test_main.py" in file_names

    # Should not find .DS_Store or __pycache__ files
    assert not any(".DS_Store" in f.file_path for f in files)
    assert not any("__pycache__" in f.file_path for f in files)


def test_scan_repository_respects_language_filter(temp_repo):
    """Test that scanner only returns configured languages."""
    config = RepositoryScanner()
    config.supported_languages = {"python"}
    scanner = RepositoryScannerImpl(config)

    files = scanner.scan_repository(temp_repo)

    # Should only have Python files
    assert all(f.language == "python" for f in files)


def test_scan_repository_metadata(scanner, temp_repo):
    """Test that file metadata is correct."""
    files = scanner.scan_repository(temp_repo)

    # Check metadata
    for file_metadata in files:
        assert file_metadata.file_path.startswith(temp_repo)
        assert file_metadata.language == "python"
        assert file_metadata.size_bytes > 0
        assert isinstance(file_metadata.modified_time, datetime)
        assert file_metadata.is_ignored is False


def test_scan_repository_not_found(scanner):
    """Test scanning non-existent repository."""
    with pytest.raises(ValueError):
        scanner.scan_repository("/nonexistent/path")


def test_scan_multiple_repositories(scanner, multi_repo):
    """Test scanning multiple repositories."""
    results = scanner.scan_multiple_repositories(multi_repo)

    assert len(results) == 2
    assert "repo1" in results
    assert "repo2" in results
    assert len(results["repo1"]) == 2
    assert len(results["repo2"]) == 1


# ==================== File Organization Tests ====================


def test_get_files_by_language(scanner, temp_repo):
    """Test organizing files by language."""
    scanner.scan_repository(temp_repo)
    files_by_lang = scanner.get_files_by_language()

    assert "python" in files_by_lang
    assert len(files_by_lang["python"]) == 3
    assert all(f.language == "python" for f in files_by_lang["python"])


def test_get_python_files(scanner, temp_repo):
    """Test getting only Python files."""
    scanner.scan_repository(temp_repo)
    python_files = scanner.get_python_files()

    assert len(python_files) == 3
    assert all(f.language == "python" for f in python_files)


def test_get_statistics(scanner, temp_repo):
    """Test getting scanning statistics."""
    scanner.scan_repository(temp_repo, "test_repo")
    stats = scanner.get_statistics()

    assert stats["total_files"] == 3
    assert stats["by_repository"]["test_repo"] == 3
    assert stats["by_language"]["python"] == 3
    assert stats["total_size_bytes"] > 0


# ==================== Configuration Management Tests ====================


def test_add_ignored_directory_to_scanner(scanner, temp_repo):
    """Test adding ignored directories after initialization."""
    # First scan
    files1 = scanner.scan_repository(temp_repo)
    initial_count = len(files1)

    # Add ignored directory
    scanner.clear_scanned_files()
    scanner.add_ignored_directory("src")

    # Scan again
    files2 = scanner.scan_repository(temp_repo)

    # Should have fewer files
    assert len(files2) < initial_count
    assert not any("src" in f.file_path for f in files2)


def test_add_supported_language(scanner, temp_repo):
    """Test adding additional supported languages."""
    # Create a non-Python file
    js_file = os.path.join(temp_repo, "script.js")
    Path(js_file).write_text("console.log('hello');\n")

    # Initially should not find JavaScript
    files1 = scanner.scan_repository(temp_repo)
    js_count1 = sum(1 for f in files1 if f.language == "javascript")
    assert js_count1 == 0

    # Add JavaScript support
    scanner.clear_scanned_files()
    scanner.add_supported_language("javascript")

    # Should now find JavaScript
    files2 = scanner.scan_repository(temp_repo)
    js_count2 = sum(1 for f in files2 if f.language == "javascript")
    assert js_count2 == 1


def test_get_repository_path(scanner, temp_repo):
    """Test retrieving repository path."""
    scanner.scan_repository(temp_repo, "my_repo")

    path = scanner.get_repository_path("my_repo")
    assert path == temp_repo

    # Non-existent repo
    path = scanner.get_repository_path("nonexistent")
    assert path is None


# ==================== Edge Cases ====================


def test_empty_repository(scanner):
    """Test scanning an empty repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = scanner.scan_repository(tmpdir)
        assert len(files) == 0


def test_repository_with_only_ignored_files(scanner):
    """Test repository containing only ignored files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create only ignored files
        Path(os.path.join(tmpdir, "file.pyc")).write_text("")
        Path(os.path.join(tmpdir, ".DS_Store")).write_text("")

        files = scanner.scan_repository(tmpdir)
        assert len(files) == 0


def test_deeply_nested_files(scanner):
    """Test scanning deeply nested directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create deeply nested structure
        deep_path = os.path.join(tmpdir, "a", "b", "c", "d", "e")
        os.makedirs(deep_path)
        Path(os.path.join(deep_path, "deep.py")).write_text("pass\n")

        files = scanner.scan_repository(tmpdir)
        assert len(files) == 1
        assert files[0].file_path.endswith("deep.py")


def test_symlink_handling(scanner):
    """Test that symlinks are handled safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create regular file
        src_file = os.path.join(tmpdir, "source.py")
        Path(src_file).write_text("print('hello')\n")

        # Create subdirectory
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)

        # Create symlink (only on systems that support it)
        try:
            link_path = os.path.join(subdir, "link.py")
            os.symlink(src_file, link_path)

            files = scanner.scan_repository(tmpdir)
            # Should find both the original and the symlink
            # (os.walk follows symlinks by default)
            assert len(files) >= 1
        except (OSError, NotImplementedError):
            # Skip on systems that don't support symlinks
            pass


def test_clear_scanned_files(scanner, temp_repo):
    """Test clearing scanned file records."""
    scanner.scan_repository(temp_repo, "repo1")
    assert len(scanner.scanned_files) == 1

    scanner.clear_scanned_files()
    assert len(scanner.scanned_files) == 0


# ==================== Integration Tests ====================


def test_full_scanning_workflow(multi_repo):
    """Test complete scanning workflow."""
    scanner = RepositoryScannerImpl()

    # Scan multiple repos
    results = scanner.scan_multiple_repositories(multi_repo)
    assert len(results) == 2

    # Get statistics
    stats = scanner.get_statistics()
    assert stats["total_files"] == 3
    assert stats["by_repository"]["repo1"] == 2
    assert stats["by_repository"]["repo2"] == 1

    # Get files by language
    files_by_lang = scanner.get_files_by_language()
    assert len(files_by_lang["python"]) == 3

    # Get all Python files
    python_files = scanner.get_python_files()
    assert len(python_files) == 3
