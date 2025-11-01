"""Batch embedding generation with caching."""

import json
import os
from datetime import datetime
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
        # Load existing cache
        self._load_cache()

        summary = {
            "total": len(items),
            "success": 0,
            "failed": 0,
            "cached": 0,
            "newly_embedded": 0
        }

        embeddings: Dict[str, Optional[List[float]]] = {}
        errors: Dict[str, str] = {}

        for item in items:
            key = self._get_cache_key(item)

            # Check if in cache
            if key in self.cache:
                embeddings[key] = self.cache[key].get("embedding")
                summary["cached"] += 1
                summary["success"] += 1
            else:
                # Embed new item
                embedding = self._embed_and_cache(item)
                embeddings[key] = embedding

                if embedding is not None:
                    summary["newly_embedded"] += 1
                    summary["success"] += 1
                else:
                    summary["failed"] += 1
                    errors[key] = f"Failed to embed {item.name}"

        # Save cache
        self._save_cache()

        return {
            "summary": summary,
            "embeddings": embeddings,
            "errors": errors,
            "metadata": self.metadata
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

    def _embed_and_cache(
        self, item: Union[Function, Class]
    ) -> Optional[List[float]]:
        """Prepare, embed, and cache a single item."""
        try:
            # Prepare text
            if isinstance(item, Function):
                text = self.text_preparator.prepare_function(item)
            else:
                text = self.text_preparator.prepare_class(item)

            # Handle empty text
            if not text or not text.strip():
                print(f"Warning: Empty text for {item.name}")
                return None

            # Generate embedding
            embedding = self.embedding_generator.embed_code(text)

            # Cache result
            key = self._get_cache_key(item)
            self.cache[key] = {
                'name': item.name,
                'embedding': embedding,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'model_version': self.metadata.get('model_version', '1.0')
            }

            return embedding
        except Exception as e:
            print(f"Error embedding {item.name}: {e}")
            return None
