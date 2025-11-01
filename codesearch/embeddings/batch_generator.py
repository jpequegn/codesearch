"""Batch embedding generation with caching."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from codesearch.models import Function, Class
from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.embeddings.text_preparator import TextPreparator


class BatchEmbeddingGenerator:
    """Generate embeddings for multiple functions/classes with caching."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        text_preparator: TextPreparator,
        cache_dir: str = "~/.codesearch/embeddings"
    ):
        """Initialize batch generator with components and cache directory."""
        self.embedding_generator = embedding_generator
        self.text_preparator = text_preparator
        self.cache_dir = os.path.expanduser(cache_dir)
        self.cache: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.cache_path = os.path.join(self.cache_dir, "cache.json")

    def process_functions(
        self, functions: List[Function]
    ) -> Dict[str, Any]:
        """Process list of functions and return embeddings."""
        return self.process_batch(functions)

    def process_classes(
        self, classes: List[Class]
    ) -> Dict[str, Any]:
        """Process list of classes and return embeddings."""
        return self.process_batch(classes)

    def process_batch(
        self, items: List[Union[Function, Class]]
    ) -> Dict[str, Any]:
        """Process mixed list of functions and classes."""
        # Placeholder - will implement in next tasks
        return {
            "summary": {"total": 0, "success": 0, "failed": 0, "cached": 0, "newly_embedded": 0},
            "embeddings": {},
            "errors": {},
            "metadata": {}
        }

    def _load_cache(self) -> None:
        """Load embeddings from disk cache into memory."""
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r') as f:
                    data = json.load(f)
                self.cache = data.get('embeddings', {})
                self.metadata = data.get('metadata', {})
            else:
                self.cache = {}
                self.metadata = self._create_metadata()
        except Exception as e:
            # Log error, continue with empty cache
            print(f"Warning: Failed to load cache: {e}")
            self.cache = {}
            self.metadata = self._create_metadata()

    def _save_cache(self) -> None:
        """Persist in-memory cache to disk."""
        try:
            self.metadata['updated'] = datetime.utcnow().isoformat() + 'Z'
            data = {
                'metadata': self.metadata,
                'embeddings': self.cache
            }
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error: Failed to save cache: {e}")

    def _create_metadata(self) -> Dict[str, Any]:
        """Create initial metadata."""
        return {
            "model_name": self.embedding_generator.model_config.name,
            "model_version": "1.0",
            "dimensions": self.embedding_generator.model_config.dimensions,
            "created": datetime.utcnow().isoformat() + 'Z',
            "updated": datetime.utcnow().isoformat() + 'Z'
        }

    def _get_cache_key(self, item: Union[Function, Class]) -> str:
        """Generate unique cache key for function/class."""
        return f"{item.file_path}:{item.line_number}"
