"""Voyage AI embedding generation for code using the Voyage Code 3 API.

Voyage Code 3 is a state-of-the-art API-based code embedding model that provides:
- 1024-dimensional embeddings (configurable via Matryoshka)
- 16,000 token context window (vs 512 for local models)
- Best-in-class code retrieval performance

Requires: pip install voyageai
"""

import logging
import os
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default constants
DEFAULT_MODEL = "voyage-code-3"
DEFAULT_DIMENSIONS = 1024
DEFAULT_MAX_LENGTH = 16000
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds


class VoyageAPIError(Exception):
    """Exception raised for Voyage API errors."""

    pass


class VoyageEmbeddingGenerator:
    """Generates embeddings using Voyage AI's Code 3 API.

    Features:
    - Rate limiting with exponential backoff
    - Batch processing for efficiency
    - Error handling and retries
    - Cost estimation

    Example:
        ```python
        generator = VoyageEmbeddingGenerator(api_key="your-key")
        embedding = generator.embed_code("def hello(): print('world')")
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        max_retries: int = MAX_RETRIES,
    ):
        """Initialize the Voyage embedding generator.

        Args:
            api_key: Voyage AI API key. If not provided, looks for
                    VOYAGE_API_KEY environment variable.
            model: Model ID to use (default: voyage-code-3)
            dimensions: Output embedding dimensions (default: 1024)
            max_retries: Maximum retry attempts for rate limiting

        Raises:
            ImportError: If voyageai package is not installed
            ValueError: If no API key is provided or found
        """
        # Resolve API key
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Voyage API key required. Set VOYAGE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.dimensions = dimensions
        self.max_retries = max_retries

        # Import voyageai (optional dependency)
        try:
            import voyageai

            self.client = voyageai.Client(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(
                "voyageai package required for Voyage embeddings. "
                "Install with: pip install voyageai"
            ) from e

        # Track API usage for cost estimation
        self._total_tokens = 0
        self._total_requests = 0

    def get_model_info(self) -> dict:
        """Return model metadata."""
        return {
            "name": self.model,
            "model_path": self.model,
            "dimensions": self.dimensions,
            "max_length": DEFAULT_MAX_LENGTH,
            "device": "api",
            "pooling": "api",  # API handles pooling
            "provider": "voyage",
        }

    def embed_code(self, code_text: str) -> List[float]:
        """Generate embedding for a single code snippet.

        Args:
            code_text: Source code as string

        Returns:
            1024-dimensional embedding vector (normalized by API)

        Raises:
            VoyageAPIError: If API request fails after retries
        """
        embeddings = self._embed_with_retry([code_text])
        return embeddings[0]

    def embed_batch(self, code_texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple code snippets.

        Args:
            code_texts: List of code snippets

        Returns:
            List of embedding vectors

        Raises:
            VoyageAPIError: If API request fails after retries
        """
        if not code_texts:
            return []

        return self._embed_with_retry(code_texts)

    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with exponential backoff retry for rate limits.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings

        Raises:
            VoyageAPIError: If all retries are exhausted
        """
        last_error = None
        backoff = INITIAL_BACKOFF

        for attempt in range(self.max_retries):
            try:
                result = self.client.embed(
                    texts,
                    model=self.model,
                    input_type="document",  # Code is treated as document
                )

                # Track usage
                self._total_requests += 1
                if hasattr(result, "total_tokens"):
                    self._total_tokens += result.total_tokens

                return result.embeddings

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if it's a rate limit error
                is_rate_limit = "rate" in error_str or "429" in error_str

                if is_rate_limit:
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            f"Rate limited, retrying in {backoff:.1f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(backoff)
                        backoff *= 2  # Exponential backoff
                        continue
                    # Last attempt for rate limit - fall through to final error
                else:
                    # For non-rate-limit errors, raise immediately
                    raise VoyageAPIError(f"Voyage API error: {e}") from e

        raise VoyageAPIError(
            f"Failed after {self.max_retries} retries: {last_error}"
        ) from last_error

    def get_usage_stats(self) -> dict:
        """Get API usage statistics.

        Returns:
            Dictionary with usage stats and estimated cost
        """
        # Voyage Code 3 costs $0.06 per 1M tokens
        cost_per_million = 0.06
        estimated_cost = (self._total_tokens / 1_000_000) * cost_per_million

        return {
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
            "estimated_cost_usd": round(estimated_cost, 4),
            "cost_per_million_tokens": cost_per_million,
        }

    def estimate_cost(self, texts: List[str]) -> dict:
        """Estimate cost for embedding a list of texts.

        Args:
            texts: List of texts to estimate

        Returns:
            Dictionary with token count and estimated cost
        """
        # Rough estimate: ~4 characters per token for code
        total_chars = sum(len(t) for t in texts)
        estimated_tokens = total_chars // 4

        cost_per_million = 0.06
        estimated_cost = (estimated_tokens / 1_000_000) * cost_per_million

        return {
            "text_count": len(texts),
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
        }
