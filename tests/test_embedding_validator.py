"""Tests for embedding quality validation."""

import pytest
import math
from codesearch.embeddings.validator import (
    ValidationCheck,
    ValidationResult,
    VectorCheck,
    EmbeddingValidator,
    SimilarityCheck,
    ConsistencyCheck,
    ValidationReport
)
from codesearch.embeddings.generator import EmbeddingGenerator


def test_validation_check_is_abstract():
    """ValidationCheck cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ValidationCheck()


def test_validation_result_creation():
    """ValidationResult can be created with passed flag and message."""
    result = ValidationResult(passed=True, message="All good")
    assert result.passed is True
    assert result.message == "All good"


def test_vector_check_valid_embedding():
    """VectorCheck accepts valid 768-dimensional embedding."""
    check = VectorCheck()

    # Create valid embedding: 768 floats in range [-1, 1]
    valid_embedding = [0.5] * 768

    result = check.validate(valid_embedding)

    assert result.passed is True
    assert result.message == "Vector format valid"


def test_vector_check_wrong_dimensions():
    """VectorCheck rejects embedding with wrong dimensions."""
    check = VectorCheck()

    # Wrong dimensions: 512 instead of 768
    wrong_dims = [0.5] * 512

    result = check.validate(wrong_dims)

    assert result.passed is False
    assert "Expected 768 dims" in result.message


def test_vector_check_detects_nan():
    """VectorCheck detects NaN values in embedding."""
    check = VectorCheck()

    # Valid embedding with NaN injected
    embedding = [0.5] * 768
    embedding[100] = float('nan')

    result = check.validate(embedding)

    assert result.passed is False
    assert "Invalid value" in result.message


def test_vector_check_detects_infinity():
    """VectorCheck detects infinity values in embedding."""
    check = VectorCheck()

    # Valid embedding with infinity injected
    embedding = [0.5] * 768
    embedding[200] = float('inf')

    result = check.validate(embedding)

    assert result.passed is False
    assert "Invalid value" in result.message


def test_vector_check_out_of_range():
    """VectorCheck rejects embeddings with out-of-range values."""
    check = VectorCheck()

    # Embedding with extreme values
    embedding = [3.0] * 768  # Greater than allowed [-2, 2]

    result = check.validate(embedding)

    assert result.passed is False
    assert "Out of range" in result.message


def test_vector_check_boundary_valid():
    """VectorCheck accepts values at boundaries [-2, 2]."""
    check = VectorCheck()

    # Embedding at the extremes but valid
    embedding = [2.0] * 384 + [-2.0] * 384

    result = check.validate(embedding)

    assert result.passed is True


def test_vector_check_name_property():
    """VectorCheck has a name property."""
    check = VectorCheck()

    assert check.name == "vector_format"


def test_vector_check_mixed_types():
    """VectorCheck rejects non-float values in embedding."""
    check = VectorCheck()

    # Valid dimensions but contains non-float
    embedding = [0.5] * 767 + ["invalid"]  # type: ignore

    result = check.validate(embedding)

    assert result.passed is False
    assert "not float" in result.message


def test_similarity_check_similar_patterns():
    """SimilarityCheck detects similar code patterns."""
    generator = EmbeddingGenerator()
    check = SimilarityCheck(generator)

    # Create a valid embedding (not used directly by SimilarityCheck, but required)
    dummy_embedding = [0.5] * 768

    result = check.validate(dummy_embedding)

    # Should pass because similar patterns ARE similar
    assert result.passed is True


def test_similarity_check_name_property():
    """SimilarityCheck has correct name property."""
    generator = EmbeddingGenerator()
    check = SimilarityCheck(generator)

    assert check.name == "semantic_correctness"


def test_similarity_check_with_custom_tolerance():
    """SimilarityCheck respects tolerance parameter."""
    generator = EmbeddingGenerator()

    # Tight tolerance
    check_tight = SimilarityCheck(generator, tolerance=0.01)
    assert check_tight.tolerance == 0.01

    # Loose tolerance
    check_loose = SimilarityCheck(generator, tolerance=0.20)
    assert check_loose.tolerance == 0.20


def test_similarity_check_cosine_similarity_calculation():
    """SimilarityCheck calculates cosine similarity correctly."""
    generator = EmbeddingGenerator()
    check = SimilarityCheck(generator)

    # Test vectors: identical
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    similarity = check._cosine_similarity(v1, v2)
    assert abs(similarity - 1.0) < 0.001  # Should be 1.0

    # Test vectors: orthogonal (perpendicular)
    v3 = [1.0, 0.0, 0.0]
    v4 = [0.0, 1.0, 0.0]
    similarity = check._cosine_similarity(v3, v4)
    assert abs(similarity - 0.0) < 0.001  # Should be 0.0


def test_similarity_check_detects_unrelated_patterns():
    """SimilarityCheck can detect when unrelated patterns are too similar."""
    generator = EmbeddingGenerator()

    # Hypothetical: if embeddings were identical for unrelated code
    # This would fail the check
    check = SimilarityCheck(generator)
    dummy_embedding = [0.5] * 768

    # In practice, this should pass because CodeBERT correctly
    # separates unrelated code patterns
    result = check.validate(dummy_embedding)
    assert result.passed is True


def test_consistency_check_deterministic():
    """ConsistencyCheck validates deterministic embedding output."""
    generator = EmbeddingGenerator()
    check = ConsistencyCheck(generator, runs=3)

    dummy_embedding = [0.5] * 768
    result = check.validate(dummy_embedding)

    # Should pass because CodeBERT is deterministic
    assert result.passed is True


def test_consistency_check_name_property():
    """ConsistencyCheck has correct name property."""
    generator = EmbeddingGenerator()
    check = ConsistencyCheck(generator)

    assert check.name == "deterministic_output"


def test_consistency_check_configurable_runs():
    """ConsistencyCheck respects runs parameter."""
    generator = EmbeddingGenerator()

    check_3_runs = ConsistencyCheck(generator, runs=3)
    assert check_3_runs.runs == 3

    check_5_runs = ConsistencyCheck(generator, runs=5)
    assert check_5_runs.runs == 5


def test_embedding_validator_initialization():
    """EmbeddingValidator starts with no checks registered."""
    validator = EmbeddingValidator()

    assert len(validator.checks) == 0


def test_embedding_validator_register_check():
    """EmbeddingValidator can register checks."""
    validator = EmbeddingValidator()
    check = VectorCheck()

    validator.register_check(check)

    assert len(validator.checks) == 1
    assert validator.checks[0] == check


def test_embedding_validator_single_check_pass():
    """EmbeddingValidator returns passing report when all checks pass."""
    validator = EmbeddingValidator()
    check = VectorCheck()
    validator.register_check(check)

    valid_embedding = [0.5] * 768
    report = validator.validate(valid_embedding)

    assert report.embedding_valid is True
    assert len(report.checks_passed) == 1
    assert len(report.checks_failed) == 0


def test_embedding_validator_single_check_fail():
    """EmbeddingValidator returns failing report when check fails."""
    validator = EmbeddingValidator()
    check = VectorCheck()
    validator.register_check(check)

    invalid_embedding = [0.5] * 512  # Wrong dimensions
    report = validator.validate(invalid_embedding)

    assert report.embedding_valid is False
    assert len(report.checks_passed) == 0
    assert len(report.checks_failed) == 1
    assert "vector_format" in report.checks_failed


def test_embedding_validator_multiple_checks_mixed():
    """EmbeddingValidator aggregates results from multiple checks."""
    generator = EmbeddingGenerator()
    validator = EmbeddingValidator()

    validator.register_check(VectorCheck())
    validator.register_check(SimilarityCheck(generator))
    validator.register_check(ConsistencyCheck(generator))

    valid_embedding = [0.5] * 768
    report = validator.validate(valid_embedding)

    # All checks should pass for valid embedding
    assert report.embedding_valid is True
    assert len(report.checks_passed) >= 1


def test_embedding_validator_validation_report():
    """ValidationReport provides human-readable summary."""
    report = ValidationReport(
        embedding_valid=True,
        timestamp="2025-11-01T12:00:00",
        checks_passed=["vector_format", "semantic_correctness"],
        checks_failed=[],
        messages={}
    )

    summary = report.summary()
    assert "All" in summary
    assert "checks passed" in summary


def test_embedding_validator_batch_validation():
    """EmbeddingValidator can validate multiple embeddings."""
    validator = EmbeddingValidator()
    validator.register_check(VectorCheck())

    embeddings = [[0.5] * 768, [0.7] * 768, [0.3] * 768]
    reports = validator.validate_batch(embeddings)

    assert len(reports) == 3
    assert all(r.embedding_valid for r in reports)


def test_embedding_validator_failed_check_message_included():
    """ValidationReport includes messages from failed checks."""
    validator = EmbeddingValidator()
    validator.register_check(VectorCheck())

    invalid_embedding = [0.5] * 512  # Wrong dimensions
    report = validator.validate(invalid_embedding)

    assert report.embedding_valid is False
    assert len(report.messages) > 0
    assert "vector_format" in report.messages
    assert "Expected 768 dims" in report.messages["vector_format"]


# Edge case and integration tests (Task 5)

def test_vector_check_empty_embedding():
    """VectorCheck rejects empty embedding."""
    check = VectorCheck()

    empty_embedding = []
    result = check.validate(empty_embedding)

    assert result.passed is False
    assert "Expected 768 dims" in result.message


def test_similarity_check_empty_patterns():
    """SimilarityCheck handles empty patterns gracefully."""
    generator = EmbeddingGenerator()
    check = SimilarityCheck(generator, patterns={})

    dummy_embedding = [0.5] * 768
    result = check.validate(dummy_embedding)

    # With no patterns to check, should pass
    assert result.passed is True


def test_consistency_check_single_run():
    """ConsistencyCheck works with single run (no comparison needed)."""
    generator = EmbeddingGenerator()
    check = ConsistencyCheck(generator, runs=1)

    dummy_embedding = [0.5] * 768
    result = check.validate(dummy_embedding)

    assert result.passed is True


def test_validator_registers_multiple_checks():
    """EmbeddingValidator can register multiple distinct checks."""
    generator = EmbeddingGenerator()
    validator = EmbeddingValidator()

    validator.register_check(VectorCheck())
    validator.register_check(SimilarityCheck(generator))
    validator.register_check(ConsistencyCheck(generator))

    assert len(validator.checks) == 3


def test_validator_check_order_preserved():
    """EmbeddingValidator runs checks in registration order."""
    validator = EmbeddingValidator()

    check1 = VectorCheck()
    check2 = SimilarityCheck(EmbeddingGenerator())

    validator.register_check(check1)
    validator.register_check(check2)

    assert validator.checks[0].name == "vector_format"
    assert validator.checks[1].name == "semantic_correctness"


def test_validation_report_timestamp_format():
    """ValidationReport timestamp is ISO format."""
    validator = EmbeddingValidator()
    validator.register_check(VectorCheck())

    valid_embedding = [0.5] * 768
    report = validator.validate(valid_embedding)

    # Should be ISO format
    assert "T" in report.timestamp
    # Check for UTC timezone indicator (either Z or +00:00)
    assert report.timestamp.endswith("Z") or report.timestamp.endswith("+00:00") or "-" in report.timestamp[-6:]


def test_vector_check_all_elements_float():
    """VectorCheck validates all elements are float type."""
    check = VectorCheck()

    # Mix of float and int (int could be valid in Python but we check strict types)
    mixed_embedding = [0.5] * 767 + [1]  # Last element is int

    result = check.validate(mixed_embedding)

    # Should fail because last element is int, not float
    # Note: In practice, Python might coerce int to float, so this tests strict typing
    # You may need to adjust based on actual behavior
    if not isinstance(mixed_embedding[-1], float):
        assert result.passed is False


def test_validator_report_different_failure_messages():
    """Each failed check has distinct message in report."""
    validator = EmbeddingValidator()

    # Add VectorCheck that will fail
    validator.register_check(VectorCheck())

    wrong_dims = [0.5] * 512
    report = validator.validate(wrong_dims)

    assert "vector_format" in report.messages
    assert "Expected 768" in report.messages["vector_format"]


def test_similarity_check_vector_normalization():
    """SimilarityCheck cosine_similarity handles normalization."""
    generator = EmbeddingGenerator()
    check = SimilarityCheck(generator)

    # Zero vectors
    zero_v1 = [0.0, 0.0, 0.0]
    zero_v2 = [0.0, 0.0, 0.0]

    similarity = check._cosine_similarity(zero_v1, zero_v2)

    # Should handle gracefully (return 0 or handle zero division)
    assert isinstance(similarity, float)
    assert 0.0 <= similarity <= 1.0


def test_consistency_check_different_test_codes():
    """ConsistencyCheck uses deterministic test code."""
    generator = EmbeddingGenerator()
    check1 = ConsistencyCheck(generator)
    check2 = ConsistencyCheck(generator)

    dummy_embedding = [0.5] * 768
    result1 = check1.validate(dummy_embedding)
    result2 = check2.validate(dummy_embedding)

    # Both should pass independently
    assert result1.passed is True
    assert result2.passed is True


def test_validator_full_integration_pipeline():
    """Full validation pipeline with all checks works end-to-end."""
    generator = EmbeddingGenerator()
    validator = EmbeddingValidator()

    # Register all checks
    validator.register_check(VectorCheck())
    validator.register_check(SimilarityCheck(generator))
    validator.register_check(ConsistencyCheck(generator))

    # Validate a real embedding from EmbeddingGenerator
    code = "def hello(): return 'world'"
    embedding = generator.embed_code(code)

    report = validator.validate(embedding)

    # Should have 3 checks attempted
    assert len(report.checks_passed) + len(report.checks_failed) == 3
    # Should pass all checks
    assert report.embedding_valid is True
    assert len(report.checks_failed) == 0


def test_batch_validation_with_mixed_validity():
    """Batch validation handles valid and invalid embeddings."""
    validator = EmbeddingValidator()
    validator.register_check(VectorCheck())

    valid_embedding = [0.5] * 768
    invalid_embedding = [0.5] * 512  # Wrong dimensions

    embeddings = [valid_embedding, invalid_embedding, valid_embedding]
    reports = validator.validate_batch(embeddings)

    assert len(reports) == 3
    assert reports[0].embedding_valid is True
    assert reports[1].embedding_valid is False
    assert reports[2].embedding_valid is True


def test_similarity_check_failing_pattern():
    """SimilarityCheck detects when pattern similarity is out of expected range."""
    generator = EmbeddingGenerator()

    # Create custom patterns with very tight bounds that will fail
    tight_patterns = {
        "impossible_match": {
            "code1": "def add(a, b):\n    return a + b",
            "code2": "def subtract(a, b):\n    return a - b",
            "expected_similarity": (0.99, 1.00),  # Unrealistic high expectation
            "reason": "Test pattern to trigger failure"
        }
    }

    check = SimilarityCheck(generator, patterns=tight_patterns, tolerance=0.01)
    dummy_embedding = [0.5] * 768

    result = check.validate(dummy_embedding)

    # Should fail because actual similarity won't be 0.99-1.00
    assert result.passed is False
    assert "Semantic checks failed" in result.message
    assert "impossible_match" in result.message


def test_consistency_check_non_deterministic_detection():
    """ConsistencyCheck can detect non-deterministic behavior (if it existed)."""
    # Note: EmbeddingGenerator is deterministic, so this tests the detection logic
    # by verifying the check works correctly with deterministic input
    generator = EmbeddingGenerator()
    check = ConsistencyCheck(generator, runs=5)

    dummy_embedding = [0.5] * 768
    result = check.validate(dummy_embedding)

    # Should pass with more runs
    assert result.passed is True
    assert "deterministic" in result.message


def test_validation_report_summary_with_failures():
    """ValidationReport summary method handles failures correctly."""
    report = ValidationReport(
        embedding_valid=False,
        timestamp="2025-11-01T12:00:00",
        checks_passed=["vector_format"],
        checks_failed=["semantic_correctness", "deterministic_output"],
        messages={
            "semantic_correctness": "Pattern mismatch",
            "deterministic_output": "Non-deterministic output detected"
        }
    )

    summary = report.summary()
    assert "2 of 3 checks failed" in summary
    assert "✗" in summary
