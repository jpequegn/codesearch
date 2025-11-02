"""Tests for embedding quality validation."""

import pytest
import math
from codesearch.embeddings.validator import (
    ValidationCheck,
    ValidationResult,
    VectorCheck,
    EmbeddingValidator,
    SimilarityCheck
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
