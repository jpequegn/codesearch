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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        self.model = AutoModel.from_pretrained(self.model_config.model_name).to(self.device)

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

        # Generate embedding using mean pooling (better than [CLS] for code)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = self._mean_pooling(outputs, inputs["attention_mask"]).cpu()

        # L2 normalize
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding[0].tolist()

    def embed_batch(self, code_texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple code snippets (batch).

        Args:
            code_texts: List of code snippets

        Returns:
            List of embedding vectors (each is 768-dimensional, normalized)
        """
        if not code_texts:
            return []

        # Tokenize all inputs at once
        inputs = self.tokenizer(
            code_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_length,
            padding=True,
        ).to(self.device)

        # Generate embeddings using mean pooling (better than [CLS] for code)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs["attention_mask"])

        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list of lists and move to CPU
        return embeddings.cpu().tolist()
