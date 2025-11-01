"""Embedding generation using transformer models."""

from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
import torch

from codesearch.models import EmbeddingModel


class EmbeddingGenerator:
    """Generates embeddings for code using HuggingFace models."""

    # Default CodeBERT model configuration
    DEFAULT_MODEL = EmbeddingModel(
        name="codebert-base",
        model_name="microsoft/codebert-base",
        dimensions=768,
        max_length=512,
        device="auto",
    )

    def __init__(self, model_config: Optional[EmbeddingModel] = None):
        """
        Initialize the embedding generator.

        Args:
            model_config: EmbeddingModel configuration (uses default CodeBERT if None)
        """
        self.model_config = model_config or self.DEFAULT_MODEL

        # Determine device
        if self.model_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.model_config.device

        # Load model and tokenizer from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name
        )
        self.model = AutoModel.from_pretrained(
            self.model_config.model_name
        ).to(self.device)

        # Set to eval mode (no gradients)
        self.model.eval()

    def get_model_info(self) -> dict:
        """Return model metadata."""
        return {
            "name": self.model_config.name,
            "model_name": self.model_config.model_name,
            "dimensions": self.model_config.dimensions,
            "max_length": self.model_config.max_length,
            "device": self.device,
        }

    def embed_code(self, code_text: str) -> List[float]:
        """
        Generate embedding for a single code snippet.

        Args:
            code_text: Source code as string

        Returns:
            768-dimensional embedding vector (normalized)
        """
        # Tokenize input
        inputs = self.tokenizer(
            code_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_length,
            padding=True,
        ).to(self.device)

        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu()

        # L2 normalize
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding[0].tolist()
