import pytest
from datetime import datetime
from unittest.mock import Mock
from codesearch.data_ingestion.audit import AuditTrail, AuditRecord


def test_audit_record_creation():
    """Test AuditRecord dataclass creation."""
    record = AuditRecord(
        batch_id="batch-123",
        timestamp=datetime.utcnow(),
        operation="insert",
        entity_id="repo:file.py:Parser:parse",
        table="code_entities"
    )

    assert record.batch_id == "batch-123"
    assert record.operation == "insert"
    assert record.entity_id == "repo:file.py:Parser:parse"
    assert record.table == "code_entities"


def test_audit_trail_initialization():
    """Test AuditTrail initialization."""
    mock_client = Mock()
    audit_trail = AuditTrail(mock_client)

    assert audit_trail.client == mock_client


def test_audit_trail_record_insertion():
    """Test recording insert operations."""
    mock_client = Mock()
    audit_trail = AuditTrail(mock_client)

    audit_trail.record_insert(
        batch_id="batch-123",
        entity_id="repo:file.py:Parser:parse",
        table="code_entities"
    )

    # Verify record was created and stored
    # (implementation will persist to database)


def test_audit_trail_rollback_retrieval():
    """Test retrieving records for rollback."""
    mock_client = Mock()
    audit_trail = AuditTrail(mock_client)

    # Record some operations
    audit_trail.record_insert("batch-123", "entity-1", "code_entities")
    audit_trail.record_insert("batch-123", "entity-2", "code_entities")

    # Get records for rollback
    records = audit_trail.get_batch_records("batch-123")

    assert len(records) == 2
    assert all(r.batch_id == "batch-123" for r in records)
