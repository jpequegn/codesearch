"""Embedding generation using transformer models."""

from typing import List, Optional, Union

import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from codesearch.embeddings.config import (
    EmbeddingConfig,
    PoolingStrategy,
    get_embedding_config,
    get_model_config,
)
from codesearch.models import EmbeddingModel


class EmbeddingGenerator:
    """Generates embeddings for code using HuggingFace models.

    Supports configuration via:
    - Explicit EmbeddingConfig or model name
    - Environment variables (CODESEARCH_MODEL, CODESEARCH_EMBEDDING_DEVICE)
    - Configuration file (~/.codesearch/config.yaml)
    - Default (UniXcoder)
    """

    def __init__(
        self,
        model_config: Optional[Union[EmbeddingConfig, EmbeddingModel, str]] = None,
        device: Optional[str] = None,
    ):
        """Initialize the embedding generator.

        Args:
            model_config: One of:
                - EmbeddingConfig: Full configuration object
                - EmbeddingModel: Legacy config object (for backwards compatibility)
                - str: Model name (e.g., "codebert", "unixcoder")
                - None: Use environment/config file/defaults
            device: Override device setting ("cpu", "cuda", "mps", "auto")
        """
        # Resolve configuration from all sources
        self.config = self._resolve_config(model_config, device)

        # Determine actual device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device

        # Load model and tokenizer from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        # CodeT5+ 770M is an encoder-decoder model, use encoder only for embeddings
        if self.config.model_name == "codet5p-770m":
            self.model = T5EncoderModel.from_pretrained(
                self.config.model_path
            ).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(
                self.config.model_path
            ).to(self.device)

        # UniXcoder-specific: add <encoder-only> token for encoder-only mode
        # This token signals the model to operate in embedding mode
        if self.config.model_name == "unixcoder":
            self._setup_unixcoder()

        # Set to eval mode (no gradients)
        self.model.eval()

    def _setup_unixcoder(self) -> None:
        """Configure tokenizer for UniXcoder encoder-only mode.

        UniXcoder uses a special <encoder-only> token to indicate that
        the model should operate in encoder-only mode for embeddings.
        This produces better quality embeddings than the default mode.
        """
        # Add the encoder-only token if not already present
        special_token = "<encoder-only>"
        if special_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([special_token], special_tokens=True)
            # Resize model embeddings to accommodate new token
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _resolve_config(
        self,
        model_config: Optional[Union[EmbeddingConfig, EmbeddingModel, str]],
        device: Optional[str],
    ) -> EmbeddingConfig:
        """Resolve configuration from various input types."""
        if isinstance(model_config, EmbeddingConfig):
            # Already an EmbeddingConfig, optionally override device
            if device:
                return EmbeddingConfig(
                    model_name=model_config.model_name,
                    model_path=model_config.model_path,
                    dimensions=model_config.dimensions,
                    max_length=model_config.max_length,
                    device=device,
                    pooling=model_config.pooling,
                    api_key=model_config.api_key,
                    api_endpoint=model_config.api_endpoint,
                )
            return model_config

        elif isinstance(model_config, EmbeddingModel):
            # Legacy EmbeddingModel - convert to EmbeddingConfig
            return EmbeddingConfig(
                model_name=model_config.name,
                model_path=model_config.model_name,
                dimensions=model_config.dimensions,
                max_length=model_config.max_length,
                device=device or model_config.device,
                pooling=PoolingStrategy.MEAN,
            )

        elif isinstance(model_config, str):
            # Model name string - look up in registry
            config = get_model_config(model_config)
            if device:
                return EmbeddingConfig(
                    model_name=config.model_name,
                    model_path=config.model_path,
                    dimensions=config.dimensions,
                    max_length=config.max_length,
                    device=device,
                    pooling=config.pooling,
                    api_key=config.api_key,
                    api_endpoint=config.api_endpoint,
                )
            return config

        else:
            # None - use environment/config file/defaults
            return get_embedding_config(model=None, device=device)

    def get_model_info(self) -> dict:
        """Return model metadata."""
        return {
            "name": self.config.model_name,
            "model_path": self.config.model_path,
            "dimensions": self.config.dimensions,
            "max_length": self.config.max_length,
            "device": self.device,
            "pooling": (
                self.config.pooling.value
                if isinstance(self.config.pooling, PoolingStrategy)
                else self.config.pooling
            ),
        }

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to token embeddings.

        Mean pooling produces more discriminative embeddings than [CLS] token
        for code similarity tasks, as it captures information from all tokens.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _get_embedding(self, model_output, attention_mask):
        """Get embedding using configured pooling strategy."""
        pooling = self.config.pooling
        if isinstance(pooling, str):
            pooling = PoolingStrategy(pooling)

        if pooling == PoolingStrategy.CLS:
            # Use [CLS] token embedding (first token)
            return model_output.last_hidden_state[:, 0, :]
        else:  # MEAN pooling (default)
            return self._mean_pooling(model_output, attention_mask)

    def embed_code(self, code_text: str) -> List[float]:
        """Generate embedding for a single code snippet.

        Args:
            code_text: Source code as string

        Returns:
            Embedding vector (dimensions depend on model, normalized)
        """
        # Tokenize input
        inputs = self.tokenizer(
            code_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        ).to(self.device)

        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = self._get_embedding(outputs, inputs["attention_mask"]).cpu()

        # L2 normalize
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding[0].tolist()

    def embed_batch(self, code_texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple code snippets (batch).

        Args:
            code_texts: List of code snippets

        Returns:
            List of embedding vectors (dimensions depend on model, normalized)
        """
        if not code_texts:
            return []

        # Tokenize all inputs at once
        inputs = self.tokenizer(
            code_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        ).to(self.device)

        # Generate embeddings for all at once
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._get_embedding(outputs, inputs["attention_mask"])

        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list of lists and move to CPU
        return embeddings.cpu().tolist()

    # Legacy property for backwards compatibility
    @property
    def model_config(self) -> EmbeddingConfig:
        """Legacy property for backwards compatibility."""
        return self.config
