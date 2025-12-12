"""MLX embedding generation for Apple Silicon.

Provides native Metal-accelerated embeddings using mlx-embedding-models library.
5-15x faster than PyTorch on M1/M2/M3/M4 chips.

Requires: pip install mlx-embedding-models
"""

import logging
import platform
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Model name mapping from our registry to mlx-embedding-models registry
MLX_MODEL_MAPPING = {
    "nomic-mlx": "nomic-text-v1.5",
    "bge-m3-mlx": "bge-m3",
    "bge-large-mlx": "bge-large",
    "bge-small-mlx": "bge-small",
}

# Default dimensions per model
MLX_MODEL_DIMENSIONS = {
    "nomic-mlx": 768,
    "bge-m3-mlx": 1024,
    "bge-large-mlx": 1024,
    "bge-small-mlx": 384,
}

# Default max lengths per model
MLX_MODEL_MAX_LENGTHS = {
    "nomic-mlx": 8192,
    "bge-m3-mlx": 8192,
    "bge-large-mlx": 512,
    "bge-small-mlx": 512,
}

# Default model
DEFAULT_MLX_MODEL = "nomic-mlx"


class MLXNotAvailableError(Exception):
    """Exception raised when MLX is not available on this platform."""

    pass


class MLXEmbeddingError(Exception):
    """Exception raised for MLX embedding errors."""

    pass


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon.

    Returns:
        True if running on Apple Silicon Mac, False otherwise.
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class MLXEmbeddingGenerator:
    """Generates embeddings using MLX on Apple Silicon.

    Features:
    - Native Metal acceleration for M1/M2/M3/M4 chips
    - 5-15x faster than PyTorch on Apple Silicon
    - Efficient batch processing
    - L2 normalized embeddings

    Example:
        ```python
        generator = MLXEmbeddingGenerator(model_name="nomic-mlx")
        embedding = generator.embed_code("def hello(): print('world')")
        ```
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MLX_MODEL,
        dimensions: Optional[int] = None,
    ) -> None:
        """Initialize MLX embedding generator.

        Args:
            model_name: Model name from our registry (e.g., "nomic-mlx")
            dimensions: Override output dimensions (model default if None)

        Raises:
            ImportError: If mlx-embedding-models not installed
            MLXNotAvailableError: If not running on Apple Silicon
            ValueError: If model_name is not supported
        """
        # Check platform
        if not is_apple_silicon():
            raise MLXNotAvailableError(
                "MLX embeddings require Apple Silicon (M1/M2/M3/M4). "
                f"Current platform: {platform.system()} {platform.machine()}"
            )

        # Validate model name
        if model_name not in MLX_MODEL_MAPPING:
            available = ", ".join(MLX_MODEL_MAPPING.keys())
            raise ValueError(
                f"Unknown MLX model '{model_name}'. Available models: {available}"
            )

        self.model_name = model_name
        self.mlx_model_id = MLX_MODEL_MAPPING[model_name]
        self.dimensions = dimensions or MLX_MODEL_DIMENSIONS[model_name]
        self.max_length = MLX_MODEL_MAX_LENGTHS[model_name]

        # Lazy import mlx-embedding-models
        try:
            from mlx_embedding_models.embedding import EmbeddingModel

            self._mlx_model_class = EmbeddingModel
        except ImportError as e:
            raise ImportError(
                "mlx-embedding-models package required for MLX embeddings. "
                "Install with: pip install mlx-embedding-models"
            ) from e

        # Load the model
        logger.info(f"Loading MLX model: {self.mlx_model_id}")
        self._model = self._mlx_model_class.from_registry(self.mlx_model_id)
        logger.info(f"MLX model loaded successfully: {self.model_name}")

    def get_model_info(self) -> dict:
        """Return model metadata.

        Returns:
            Dictionary with model information.
        """
        return {
            "name": self.model_name,
            "model_path": self.mlx_model_id,
            "dimensions": self.dimensions,
            "max_length": self.max_length,
            "device": "mlx",
            "pooling": "mean",
            "provider": "mlx-embedding-models",
        }

    def embed_code(self, code_text: str) -> List[float]:
        """Generate embedding for a single code snippet.

        Args:
            code_text: Source code as string

        Returns:
            Embedding vector (L2 normalized)

        Raises:
            MLXEmbeddingError: If embedding generation fails
        """
        try:
            embeddings = self._model.encode([code_text])
            # L2 normalize the embedding
            normalized = self._l2_normalize(embeddings[0])
            return normalized.tolist()
        except Exception as e:
            raise MLXEmbeddingError(f"Failed to generate embedding: {e}") from e

    def embed_batch(self, code_texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple code snippets.

        Args:
            code_texts: List of code snippets

        Returns:
            List of embedding vectors (L2 normalized)

        Raises:
            MLXEmbeddingError: If embedding generation fails
        """
        if not code_texts:
            return []

        try:
            embeddings = self._model.encode(code_texts)
            # L2 normalize each embedding
            return [self._l2_normalize(emb).tolist() for emb in embeddings]
        except Exception as e:
            raise MLXEmbeddingError(f"Failed to generate batch embeddings: {e}") from e

    def _l2_normalize(
        self, vector: Union["np.ndarray", List[float]]
    ) -> "np.ndarray":
        """L2 normalize a vector using numpy-compatible operations.

        Args:
            vector: Input vector (numpy array or MLX array)

        Returns:
            L2 normalized vector
        """
        import numpy as np

        # Convert to numpy if needed
        if hasattr(vector, "tolist"):
            arr = np.array(vector.tolist()) if not isinstance(vector, np.ndarray) else vector
        else:
            arr = np.array(vector)

        # Compute L2 norm
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr
