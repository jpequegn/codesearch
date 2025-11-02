"""Embedding generator for code snippets."""

from typing import List
import hashlib
import re


class EmbeddingGenerator:
    """Generates embeddings for code snippets.

    This is a minimal stub implementation for testing purposes.
    The actual implementation will use CodeBERT or similar models.
    """

    def __init__(self):
        """Initialize the embedding generator."""
        # Pattern signatures for semantic similarity
        self.pattern_signatures = {
            # Arithmetic operations share high similarity
            'arithmetic': ['add', 'subtract', 'multiply', 'divide', '+', '-', '*', '/', 'return a'],
            # Conditional logic shares similarity
            'conditional': ['if', 'else', 'elif', 'return True', 'return False', '>', '<', '>=', '<='],
            # Parsing operations
            'parsing': ['parse', 'json', 'loads', 'dumps', 'text'],
            # Math operations
            'math': ['factorial', 'fibonacci', 'sqrt', 'pow', '**'],
            # Network operations
            'network': ['http', 'request', 'get', 'post', 'url', 'requests'],
        }

    def embed_code(self, code: str) -> List[float]:
        """Generate a 768-dimensional embedding for code.

        For testing purposes, this creates a deterministic embedding based on
        code patterns. Similar code patterns will produce similar embeddings.

        Args:
            code: Source code string to embed

        Returns:
            768-dimensional embedding vector
        """
        # Normalize code (lowercase, remove extra whitespace)
        normalized = ' '.join(code.lower().split())

        # Detect pattern signatures
        pattern_scores = {}
        for pattern_name, keywords in self.pattern_signatures.items():
            score = sum(1 for kw in keywords if kw in normalized)
            pattern_scores[pattern_name] = score

        # Create base embedding from hash
        hash_obj = hashlib.md5(normalized.encode())
        hash_bytes = hash_obj.digest()

        # Initialize embedding
        embedding = []

        # Distribute pattern signals across dimensions
        # First 512 dimensions: pattern-based
        # Last 256 dimensions: hash-based for uniqueness
        for i in range(512):
            # Cycle through patterns
            patterns = list(pattern_scores.keys())
            pattern_idx = i % len(patterns)
            pattern_name = patterns[pattern_idx]
            pattern_score = pattern_scores[pattern_name]

            # Score-based value with some noise from hash
            byte_val = hash_bytes[i % 16]
            noise = (byte_val / 255.0) * 0.2  # Small noise 0-0.2
            signal = pattern_score / 10.0  # Pattern signal 0-1

            # Combine signal and noise
            float_val = signal + noise - 0.5  # Center around 0
            embedding.append(float_val)

        # Last 256 dimensions: hash-based for uniqueness
        for i in range(256):
            byte_idx = i % 16
            byte_val = hash_bytes[byte_idx]
            float_val = (byte_val / 127.5) - 1.0
            embedding.append(float_val)

        return embedding
