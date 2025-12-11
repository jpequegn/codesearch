"""Configuration system for embedding models.

Supports configuration from multiple sources with priority:
1. Explicit function arguments
2. CLI flags (--model)
3. Environment variables (CODESEARCH_MODEL, CODESEARCH_EMBEDDING_DEVICE)
4. Configuration file (~/.codesearch/config.yaml)
5. Default values
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from codesearch.cli.config import get_config_file, load_config_file


class PoolingStrategy(str, Enum):
    """Pooling strategy for generating embeddings from token representations."""
    MEAN = "mean"  # Average of all token embeddings (default, better for code)
    CLS = "cls"    # Use [CLS] token embedding


@dataclass
class EmbeddingConfig:
    """Configuration for an embedding model.

    Attributes:
        model_name: Short name for the model (e.g., "codebert", "unixcoder")
        model_path: HuggingFace model ID or path (e.g., "microsoft/codebert-base")
        dimensions: Output embedding dimensions
        max_length: Maximum input sequence length in tokens
        device: Compute device ("auto", "cpu", "cuda", "mps")
        pooling: Pooling strategy for generating embeddings
        api_key: Optional API key for API-based models (e.g., voyage-code)
        api_endpoint: Optional API endpoint for API-based models
    """
    model_name: str
    model_path: str
    dimensions: int
    max_length: int
    device: str = "auto"
    pooling: PoolingStrategy = PoolingStrategy.MEAN
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert config to dictionary for storage."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "dimensions": self.dimensions,
            "max_length": self.max_length,
            "device": self.device,
            "pooling": (
                self.pooling.value if isinstance(self.pooling, PoolingStrategy)
                else self.pooling
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EmbeddingConfig":
        """Create config from dictionary."""
        pooling = data.get("pooling", "mean")
        if isinstance(pooling, str):
            pooling = PoolingStrategy(pooling)

        return cls(
            model_name=data["model_name"],
            model_path=data["model_path"],
            dimensions=data["dimensions"],
            max_length=data["max_length"],
            device=data.get("device", "auto"),
            pooling=pooling,
            api_key=data.get("api_key"),
            api_endpoint=data.get("api_endpoint"),
        )


# Model Registry - supported embedding models
MODEL_REGISTRY: Dict[str, EmbeddingConfig] = {
    # Microsoft CodeBERT - default, well-tested for code
    "codebert": EmbeddingConfig(
        model_name="codebert",
        model_path="microsoft/codebert-base",
        dimensions=768,
        max_length=512,
        pooling=PoolingStrategy.MEAN,
    ),

    # Microsoft UniXcoder - better cross-lingual code understanding
    "unixcoder": EmbeddingConfig(
        model_name="unixcoder",
        model_path="microsoft/unixcoder-base",
        dimensions=768,
        max_length=512,
        pooling=PoolingStrategy.MEAN,
    ),

    # Salesforce CodeT5+ 110M - encoder-decoder with strong code understanding
    "codet5p-110m": EmbeddingConfig(
        model_name="codet5p-110m",
        model_path="Salesforce/codet5p-110m-embedding",
        dimensions=256,
        max_length=512,
        pooling=PoolingStrategy.MEAN,
    ),

    # Salesforce CodeT5+ 220M - larger version
    "codet5p-220m": EmbeddingConfig(
        model_name="codet5p-220m",
        model_path="Salesforce/codet5p-220m-embedding",
        dimensions=768,
        max_length=512,
        pooling=PoolingStrategy.MEAN,
    ),

    # Salesforce CodeT5+ 770M - largest version, GPU recommended
    # Uses encoder-only for embeddings from encoder-decoder model
    "codet5p-770m": EmbeddingConfig(
        model_name="codet5p-770m",
        model_path="Salesforce/codet5p-770m",
        dimensions=1024,
        max_length=512,
        pooling=PoolingStrategy.MEAN,
    ),

    # GraphCodeBERT - code + data flow aware
    "graphcodebert": EmbeddingConfig(
        model_name="graphcodebert",
        model_path="microsoft/graphcodebert-base",
        dimensions=768,
        max_length=512,
        pooling=PoolingStrategy.MEAN,
    ),

    # Voyage Code 3 - API-based, state-of-the-art code embeddings
    # Requires VOYAGE_API_KEY environment variable
    "voyage-code-3": EmbeddingConfig(
        model_name="voyage-code-3",
        model_path="voyage-code-3",  # API model ID
        dimensions=1024,
        max_length=16000,  # 16K token context
        device="api",
        pooling=PoolingStrategy.MEAN,  # API handles pooling
        api_endpoint="https://api.voyageai.com/v1/embeddings",
    ),

    # =========================================================================
    # MLX Models - Apple Silicon optimized (requires mlx-embedding-models)
    # =========================================================================

    # Nomic Text v1.5 - Best general-purpose MLX model
    "nomic-mlx": EmbeddingConfig(
        model_name="nomic-mlx",
        model_path="nomic-text-v1.5",  # mlx-embedding-models registry name
        dimensions=768,
        max_length=8192,
        device="mlx",
        pooling=PoolingStrategy.MEAN,
    ),

    # BGE-M3 - Highest quality, multi-lingual
    "bge-m3-mlx": EmbeddingConfig(
        model_name="bge-m3-mlx",
        model_path="bge-m3",
        dimensions=1024,
        max_length=8192,
        device="mlx",
        pooling=PoolingStrategy.MEAN,
    ),

    # BGE Large - Strong baseline
    "bge-large-mlx": EmbeddingConfig(
        model_name="bge-large-mlx",
        model_path="bge-large",
        dimensions=1024,
        max_length=512,
        device="mlx",
        pooling=PoolingStrategy.MEAN,
    ),

    # BGE Small - Fastest, smaller dimensions
    "bge-small-mlx": EmbeddingConfig(
        model_name="bge-small-mlx",
        model_path="bge-small",
        dimensions=384,
        max_length=512,
        device="mlx",
        pooling=PoolingStrategy.MEAN,
    ),
}

# Default model name - UniXcoder provides better embeddings than CodeBERT
# while maintaining the same size (125M params) and CPU-friendliness
DEFAULT_MODEL_NAME = "unixcoder"

# MLX models set for easy checking
MLX_MODELS = {"nomic-mlx", "bge-m3-mlx", "bge-large-mlx", "bge-small-mlx"}

# Default MLX model for Apple Silicon users
DEFAULT_MLX_MODEL = "nomic-mlx"


def is_mlx_model(model_name: str) -> bool:
    """Check if a model requires the MLX backend.

    Args:
        model_name: Name of the model to check

    Returns:
        True if the model uses MLX backend
    """
    return model_name in MLX_MODELS or model_name.endswith("-mlx")


def get_available_models() -> list[str]:
    """Get list of available model names."""
    return list(MODEL_REGISTRY.keys())


def get_model_config(model_name: str) -> EmbeddingConfig:
    """Get configuration for a registered model.

    Args:
        model_name: Name of the model (e.g., "codebert", "unixcoder")

    Returns:
        EmbeddingConfig for the model

    Raises:
        ValueError: If model is not in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
    return MODEL_REGISTRY[model_name]


def get_embedding_config(
    model: Optional[str] = None,
    device: Optional[str] = None,
) -> EmbeddingConfig:
    """Get embedding configuration from all sources.

    Priority (highest to lowest):
    1. Explicit arguments (model, device)
    2. Environment variables (CODESEARCH_MODEL, CODESEARCH_EMBEDDING_DEVICE)
    3. Configuration file embedding section
    4. Default values

    Special handling for MLX:
    - If device="mlx" and model is not an MLX model, use DEFAULT_MLX_MODEL
    - MLX models automatically use device="mlx"

    Args:
        model: Model name override
        device: Device override

    Returns:
        EmbeddingConfig with resolved values
    """
    # Start with determining the model name
    resolved_model = model
    resolved_device = device

    # Check environment variables
    if resolved_model is None:
        resolved_model = os.environ.get("CODESEARCH_MODEL")
    if resolved_device is None:
        resolved_device = os.environ.get("CODESEARCH_EMBEDDING_DEVICE")

    # Check config file
    config_file = get_config_file()
    file_config = {}
    if config_file:
        try:
            full_config = load_config_file(config_file)
            file_config = full_config.get("embedding", {})
        except Exception:
            pass  # Ignore config file errors

    if resolved_model is None:
        resolved_model = file_config.get("model")
    if resolved_device is None:
        resolved_device = file_config.get("device")

    # Handle MLX device: if user requests mlx device but doesn't specify
    # an MLX model, use the default MLX model
    if resolved_device == "mlx" and (resolved_model is None or not is_mlx_model(resolved_model)):
        resolved_model = DEFAULT_MLX_MODEL

    # Use defaults if still not set
    if resolved_model is None:
        resolved_model = DEFAULT_MODEL_NAME

    # Get the model config from registry
    config = get_model_config(resolved_model)

    # Determine final device:
    # - If user explicitly specified device, use it
    # - If device is not specified (None) or "auto", use model's defined device
    if resolved_device is None or resolved_device == "auto":
        # Use model's defined device (respects MLX, API, etc.)
        pass
    elif resolved_device != config.device:
        # Create a new config with the overridden device
        config = EmbeddingConfig(
            model_name=config.model_name,
            model_path=config.model_path,
            dimensions=config.dimensions,
            max_length=config.max_length,
            device=resolved_device,
            pooling=config.pooling,
            api_key=config.api_key,
            api_endpoint=config.api_endpoint,
        )

    # Check for API key from environment for API-based models
    # Try model-specific key first, then generic key
    api_key = None
    if config.model_name == "voyage-code-3":
        api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        api_key = os.environ.get("CODESEARCH_EMBEDDING_API_KEY")

    if api_key and config.api_key is None:
        config = EmbeddingConfig(
            model_name=config.model_name,
            model_path=config.model_path,
            dimensions=config.dimensions,
            max_length=config.max_length,
            device=config.device,
            pooling=config.pooling,
            api_key=api_key,
            api_endpoint=config.api_endpoint,
        )

    return config
