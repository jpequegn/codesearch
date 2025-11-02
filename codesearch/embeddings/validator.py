"""Embedding quality validation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
import math


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    passed: bool
    message: str


class ValidationCheck(ABC):
    """Base class for all embedding validation checks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this check."""
        pass

    @abstractmethod
    def validate(self, embedding: List[float]) -> ValidationResult:
        """Run validation on an embedding.

        Args:
            embedding: 768-dimensional embedding vector

        Returns:
            ValidationResult with passed flag and message
        """
        pass


class VectorCheck(ValidationCheck):
    """Validates embedding vector format and properties."""

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

    @property
    def name(self) -> str:
        return "vector_format"

    def validate(self, embedding: List[float]) -> ValidationResult:
        """Check dimensions, detect NaN/infinity, validate value ranges."""
        # Check 1: Dimensions must be 768 for CodeBERT
        if len(embedding) != self.dimensions:
            return ValidationResult(
                passed=False,
                message=f"Expected {self.dimensions} dims, got {len(embedding)}"
            )

        # Check 2: Detect NaN/Infinity and type validation
        for i, val in enumerate(embedding):
            if not isinstance(val, float):
                return ValidationResult(
                    passed=False,
                    message=f"Element {i} not float: {type(val)}"
                )
            if math.isnan(val) or math.isinf(val):
                return ValidationResult(
                    passed=False,
                    message=f"Invalid value at index {i}: {val}"
                )

        # Check 3: Value range (after normalization, typically [-1, 1])
        min_val = min(embedding)
        max_val = max(embedding)
        if not (-2.0 <= min_val <= 2.0 and -2.0 <= max_val <= 2.0):
            return ValidationResult(
                passed=False,
                message=f"Out of range: min={min_val:.3f}, max={max_val:.3f}"
            )

        return ValidationResult(passed=True, message="Vector format valid")


class EmbeddingValidator:
    """Orchestrates validation checks and aggregates results."""

    def __init__(self):
        self.checks: List[ValidationCheck] = []

    def register_check(self, check: ValidationCheck) -> None:
        """Register a validation check.

        Args:
            check: ValidationCheck instance to add
        """
        self.checks.append(check)

    def validate(self, embedding: List[float]) -> Dict[str, bool]:
        """Run all registered checks on an embedding.

        Args:
            embedding: 768-dimensional embedding vector

        Returns:
            Dictionary with validation results
        """
        results = {}
        for check in self.checks:
            result = check.validate(embedding)
            results[check.name] = result.passed
        return results
