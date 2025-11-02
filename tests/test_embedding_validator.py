"""Tests for embedding quality validation."""

import pytest
import math
from codesearch.embeddings.validator import (
    ValidationCheck,
    ValidationResult,
    VectorCheck,
    EmbeddingValidator
)


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
