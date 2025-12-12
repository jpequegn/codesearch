"""Tests for MLX embedding generator."""

import platform
import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from codesearch.embeddings.mlx import (
    DEFAULT_MLX_MODEL,
    MLX_MODEL_DIMENSIONS,
    MLX_MODEL_MAPPING,
    MLX_MODEL_MAX_LENGTHS,
    MLXEmbeddingError,
    MLXNotAvailableError,
    is_apple_silicon,
)


class TestIsAppleSilicon:
    """Tests for is_apple_silicon helper function."""

    def test_apple_silicon_detection_mac_arm64(self) -> None:
        """Test detection on Apple Silicon Mac."""
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            assert is_apple_silicon() is True

    def test_apple_silicon_detection_mac_x86(self) -> None:
        """Test detection on Intel Mac."""
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="x86_64"),
        ):
            assert is_apple_silicon() is False

    def test_apple_silicon_detection_linux(self) -> None:
        """Test detection on Linux."""
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
        ):
            assert is_apple_silicon() is False

    def test_apple_silicon_detection_windows(self) -> None:
        """Test detection on Windows."""
        with (
            patch("platform.system", return_value="Windows"),
            patch("platform.machine", return_value="AMD64"),
        ):
            assert is_apple_silicon() is False


class TestMLXEmbeddingGeneratorInit:
    """Tests for MLXEmbeddingGenerator initialization."""

    def test_init_fails_on_non_apple_silicon(self) -> None:
        """Test that init raises error on non-Apple Silicon."""
        with patch("codesearch.embeddings.mlx.is_apple_silicon", return_value=False):
            with pytest.raises(MLXNotAvailableError) as exc_info:
                from codesearch.embeddings.mlx import MLXEmbeddingGenerator
                MLXEmbeddingGenerator()

            assert "Apple Silicon" in str(exc_info.value)

    def test_init_fails_with_unknown_model(self) -> None:
        """Test that init raises error for unknown model."""
        # Create mock for mlx_embedding_models
        mock_module = MagicMock()
        mock_model = MagicMock()
        mock_module.embedding.EmbeddingModel = MagicMock(return_value=mock_model)

        with (
            patch("codesearch.embeddings.mlx.is_apple_silicon", return_value=True),
            patch.dict(sys.modules, {
                "mlx_embedding_models": mock_module,
                "mlx_embedding_models.embedding": mock_module.embedding,
            }),
        ):
            with pytest.raises(ValueError) as exc_info:
                from codesearch.embeddings.mlx import MLXEmbeddingGenerator
                MLXEmbeddingGenerator(model_name="unknown-model")

            assert "Unknown MLX model" in str(exc_info.value)

    def test_init_fails_without_mlx_embedding_models(self) -> None:
        """Test that init raises ImportError when mlx-embedding-models not installed."""
        with (
            patch("codesearch.embeddings.mlx.is_apple_silicon", return_value=True),
            patch.dict(sys.modules, {"mlx_embedding_models": None}),
        ):
            # Force reimport to trigger ImportError
            import codesearch.embeddings.mlx as mlx_module

            # Patch the import inside the class
            def mock_init(
                self: MagicMock,
                model_name: str = DEFAULT_MLX_MODEL,
                dimensions: int | None = None,
            ) -> None:
                if not is_apple_silicon():
                    raise MLXNotAvailableError("Not Apple Silicon")
                if model_name not in MLX_MODEL_MAPPING:
                    raise ValueError("Unknown model")
                raise ImportError("mlx-embedding-models not installed")

            with (
                patch.object(mlx_module.MLXEmbeddingGenerator, "__init__", mock_init),
                pytest.raises(ImportError),
            ):
                mlx_module.MLXEmbeddingGenerator()

    def test_init_success_with_default_model(self) -> None:
        """Test successful initialization with default model."""
        mock_embedding_model = MagicMock()
        mock_embedding_model.from_registry = MagicMock(return_value=MagicMock())

        mock_module = MagicMock()
        mock_module.embedding.EmbeddingModel = mock_embedding_model

        with (
            patch("codesearch.embeddings.mlx.is_apple_silicon", return_value=True),
            patch.dict(sys.modules, {
                "mlx_embedding_models": mock_module,
                "mlx_embedding_models.embedding": mock_module.embedding,
            }),
        ):
            from codesearch.embeddings.mlx import MLXEmbeddingGenerator
            generator = MLXEmbeddingGenerator()

            assert generator.model_name == DEFAULT_MLX_MODEL
            assert generator.dimensions == MLX_MODEL_DIMENSIONS[DEFAULT_MLX_MODEL]
            assert generator.max_length == MLX_MODEL_MAX_LENGTHS[DEFAULT_MLX_MODEL]

    def test_init_with_custom_dimensions(self) -> None:
        """Test initialization with custom dimensions."""
        mock_embedding_model = MagicMock()
        mock_embedding_model.from_registry = MagicMock(return_value=MagicMock())

        mock_module = MagicMock()
        mock_module.embedding.EmbeddingModel = mock_embedding_model

        with (
            patch("codesearch.embeddings.mlx.is_apple_silicon", return_value=True),
            patch.dict(sys.modules, {
                "mlx_embedding_models": mock_module,
                "mlx_embedding_models.embedding": mock_module.embedding,
            }),
        ):
            from codesearch.embeddings.mlx import MLXEmbeddingGenerator
            generator = MLXEmbeddingGenerator(dimensions=512)

            assert generator.dimensions == 512

    def test_init_with_each_supported_model(self) -> None:
        """Test initialization with each supported model."""
        mock_embedding_model = MagicMock()
        mock_embedding_model.from_registry = MagicMock(return_value=MagicMock())

        mock_module = MagicMock()
        mock_module.embedding.EmbeddingModel = mock_embedding_model

        for model_name in MLX_MODEL_MAPPING:
            with (
                patch("codesearch.embeddings.mlx.is_apple_silicon", return_value=True),
                patch.dict(sys.modules, {
                    "mlx_embedding_models": mock_module,
                    "mlx_embedding_models.embedding": mock_module.embedding,
                }),
            ):
                from codesearch.embeddings.mlx import MLXEmbeddingGenerator
                generator = MLXEmbeddingGenerator(model_name=model_name)

                assert generator.model_name == model_name
                assert generator.mlx_model_id == MLX_MODEL_MAPPING[model_name]


class TestMLXEmbeddingGeneratorMethods:
    """Tests for MLXEmbeddingGenerator methods using mocks."""

    @pytest.fixture
    def mock_generator(self) -> Generator[MagicMock, None, None]:
        """Create a mock generator for testing methods."""
        mock_embedding_model = MagicMock()
        mock_model_instance = MagicMock()
        mock_embedding_model.from_registry = MagicMock(return_value=mock_model_instance)

        mock_module = MagicMock()
        mock_module.embedding.EmbeddingModel = mock_embedding_model

        with (
            patch("codesearch.embeddings.mlx.is_apple_silicon", return_value=True),
            patch.dict(sys.modules, {
                "mlx_embedding_models": mock_module,
                "mlx_embedding_models.embedding": mock_module.embedding,
            }),
        ):
            from codesearch.embeddings.mlx import MLXEmbeddingGenerator
            generator = MLXEmbeddingGenerator()
            generator._model = mock_model_instance
            yield generator

    def test_get_model_info(self, mock_generator: MagicMock) -> None:
        """Test get_model_info returns correct metadata."""
        info = mock_generator.get_model_info()

        assert info["name"] == DEFAULT_MLX_MODEL
        assert info["model_path"] == MLX_MODEL_MAPPING[DEFAULT_MLX_MODEL]
        assert info["dimensions"] == MLX_MODEL_DIMENSIONS[DEFAULT_MLX_MODEL]
        assert info["device"] == "mlx"
        assert info["provider"] == "mlx-embedding-models"

    def test_embed_code_returns_normalized_vector(self, mock_generator: MagicMock) -> None:
        """Test embed_code returns L2 normalized vector."""
        # Create a non-normalized vector
        raw_embedding = np.array([3.0, 4.0])  # L2 norm = 5
        mock_generator._model.encode = MagicMock(return_value=[raw_embedding])

        result = mock_generator.embed_code("def test(): pass")

        # Check it's normalized (norm should be ~1.0)
        result_array = np.array(result)
        norm = np.linalg.norm(result_array)
        assert abs(norm - 1.0) < 0.0001

        # Check values are correct [0.6, 0.8]
        assert abs(result[0] - 0.6) < 0.0001
        assert abs(result[1] - 0.8) < 0.0001

    def test_embed_batch_returns_list_of_normalized_vectors(
        self, mock_generator: MagicMock
    ) -> None:
        """Test embed_batch returns list of L2 normalized vectors."""
        raw_embeddings = [
            np.array([3.0, 4.0]),   # norm = 5 -> [0.6, 0.8]
            np.array([1.0, 0.0]),   # norm = 1 -> [1.0, 0.0]
        ]
        mock_generator._model.encode = MagicMock(return_value=raw_embeddings)

        results = mock_generator.embed_batch(["code1", "code2"])

        assert len(results) == 2

        # Check first embedding is normalized
        norm1 = np.linalg.norm(np.array(results[0]))
        assert abs(norm1 - 1.0) < 0.0001

        # Check second embedding is normalized
        norm2 = np.linalg.norm(np.array(results[1]))
        assert abs(norm2 - 1.0) < 0.0001

    def test_embed_batch_empty_list(self, mock_generator: MagicMock) -> None:
        """Test embed_batch handles empty list."""
        result = mock_generator.embed_batch([])
        assert result == []

    def test_embed_code_error_handling(self, mock_generator: MagicMock) -> None:
        """Test embed_code raises MLXEmbeddingError on failure."""
        mock_generator._model.encode = MagicMock(side_effect=RuntimeError("Model error"))

        with pytest.raises(MLXEmbeddingError) as exc_info:
            mock_generator.embed_code("def test(): pass")

        assert "Failed to generate embedding" in str(exc_info.value)

    def test_embed_batch_error_handling(self, mock_generator: MagicMock) -> None:
        """Test embed_batch raises MLXEmbeddingError on failure."""
        mock_generator._model.encode = MagicMock(side_effect=RuntimeError("Model error"))

        with pytest.raises(MLXEmbeddingError) as exc_info:
            mock_generator.embed_batch(["code1", "code2"])

        assert "Failed to generate batch embeddings" in str(exc_info.value)


class TestL2Normalization:
    """Tests for L2 normalization."""

    def test_l2_normalize_basic(self) -> None:
        """Test L2 normalization with basic vector."""
        mock_embedding_model = MagicMock()
        mock_model_instance = MagicMock()
        mock_embedding_model.from_registry = MagicMock(return_value=mock_model_instance)

        mock_module = MagicMock()
        mock_module.embedding.EmbeddingModel = mock_embedding_model

        with (
            patch("codesearch.embeddings.mlx.is_apple_silicon", return_value=True),
            patch.dict(sys.modules, {
                "mlx_embedding_models": mock_module,
                "mlx_embedding_models.embedding": mock_module.embedding,
            }),
        ):
            from codesearch.embeddings.mlx import MLXEmbeddingGenerator
            generator = MLXEmbeddingGenerator()

            vector = np.array([3.0, 4.0])
            normalized = generator._l2_normalize(vector)

            # Expected: [0.6, 0.8]
            assert abs(normalized[0] - 0.6) < 0.0001
            assert abs(normalized[1] - 0.8) < 0.0001

    def test_l2_normalize_zero_vector(self) -> None:
        """Test L2 normalization with zero vector."""
        mock_embedding_model = MagicMock()
        mock_model_instance = MagicMock()
        mock_embedding_model.from_registry = MagicMock(return_value=mock_model_instance)

        mock_module = MagicMock()
        mock_module.embedding.EmbeddingModel = mock_embedding_model

        with (
            patch("codesearch.embeddings.mlx.is_apple_silicon", return_value=True),
            patch.dict(sys.modules, {
                "mlx_embedding_models": mock_module,
                "mlx_embedding_models.embedding": mock_module.embedding,
            }),
        ):
            from codesearch.embeddings.mlx import MLXEmbeddingGenerator
            generator = MLXEmbeddingGenerator()

            vector = np.array([0.0, 0.0])
            normalized = generator._l2_normalize(vector)

            # Should not divide by zero, return zero vector
            assert normalized[0] == 0.0
            assert normalized[1] == 0.0

    def test_l2_normalize_already_normalized(self) -> None:
        """Test L2 normalization with already normalized vector."""
        mock_embedding_model = MagicMock()
        mock_model_instance = MagicMock()
        mock_embedding_model.from_registry = MagicMock(return_value=mock_model_instance)

        mock_module = MagicMock()
        mock_module.embedding.EmbeddingModel = mock_embedding_model

        with (
            patch("codesearch.embeddings.mlx.is_apple_silicon", return_value=True),
            patch.dict(sys.modules, {
                "mlx_embedding_models": mock_module,
                "mlx_embedding_models.embedding": mock_module.embedding,
            }),
        ):
            from codesearch.embeddings.mlx import MLXEmbeddingGenerator
            generator = MLXEmbeddingGenerator()

            vector = np.array([0.6, 0.8])  # Already normalized
            normalized = generator._l2_normalize(vector)

            # Should stay the same
            assert abs(normalized[0] - 0.6) < 0.0001
            assert abs(normalized[1] - 0.8) < 0.0001


class TestMLXModelConstants:
    """Tests for MLX model constants."""

    def test_model_mapping_complete(self) -> None:
        """Test that all expected models are in mapping."""
        expected = {"nomic-mlx", "bge-m3-mlx", "bge-large-mlx", "bge-small-mlx"}
        assert set(MLX_MODEL_MAPPING.keys()) == expected

    def test_model_dimensions_complete(self) -> None:
        """Test that all models have dimensions defined."""
        for model in MLX_MODEL_MAPPING:
            assert model in MLX_MODEL_DIMENSIONS

    def test_model_max_lengths_complete(self) -> None:
        """Test that all models have max lengths defined."""
        for model in MLX_MODEL_MAPPING:
            assert model in MLX_MODEL_MAX_LENGTHS

    def test_default_model_is_valid(self) -> None:
        """Test that default model is in mapping."""
        assert DEFAULT_MLX_MODEL in MLX_MODEL_MAPPING


# Integration test - only runs on Apple Silicon
@pytest.mark.skipif(
    not (platform.system() == "Darwin" and platform.machine() == "arm64"),
    reason="Requires Apple Silicon"
)
class TestMLXIntegration:
    """Integration tests that run on actual Apple Silicon hardware."""

    @pytest.mark.skip(reason="Requires mlx-embedding-models package installed")
    def test_real_embedding_generation(self) -> None:
        """Test actual embedding generation on Apple Silicon."""
        from codesearch.embeddings.mlx import MLXEmbeddingGenerator

        generator = MLXEmbeddingGenerator(model_name="bge-small-mlx")

        embedding = generator.embed_code("def hello(): print('world')")

        assert len(embedding) == 384  # bge-small dimensions
        assert abs(np.linalg.norm(np.array(embedding)) - 1.0) < 0.0001  # Normalized

    @pytest.mark.skip(reason="Requires mlx-embedding-models package installed")
    def test_real_batch_embedding(self) -> None:
        """Test actual batch embedding on Apple Silicon."""
        from codesearch.embeddings.mlx import MLXEmbeddingGenerator

        generator = MLXEmbeddingGenerator(model_name="bge-small-mlx")

        embeddings = generator.embed_batch([
            "def hello(): print('world')",
            "class Foo: pass",
        ])

        assert len(embeddings) == 2
        for emb in embeddings:
            assert len(emb) == 384
            assert abs(np.linalg.norm(np.array(emb)) - 1.0) < 0.0001
