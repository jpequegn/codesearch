"""Tests for EmbeddingGenerator with MLX routing."""

from unittest.mock import MagicMock, patch


class TestMLXRouting:
    """Tests for MLX model routing in EmbeddingGenerator."""

    def test_generator_routes_to_mlx_by_model_name(self) -> None:
        """Test that MLX models route to MLXEmbeddingGenerator."""
        mock_mlx_generator = MagicMock()
        mock_mlx_class = MagicMock(return_value=mock_mlx_generator)

        with patch(
            "codesearch.embeddings.mlx.MLXEmbeddingGenerator", mock_mlx_class
        ):
            from codesearch.embeddings.generator import EmbeddingGenerator

            generator = EmbeddingGenerator(model_config="nomic-mlx")

            assert generator._mlx_generator is not None
            assert generator._api_generator is None
            assert generator.device == "mlx"
            mock_mlx_class.assert_called_once()

    def test_generator_routes_to_mlx_by_device(self) -> None:
        """Test that device=mlx routes to MLXEmbeddingGenerator."""
        mock_mlx_generator = MagicMock()
        mock_mlx_class = MagicMock(return_value=mock_mlx_generator)

        with patch(
            "codesearch.embeddings.mlx.MLXEmbeddingGenerator", mock_mlx_class
        ):
            from codesearch.embeddings.generator import EmbeddingGenerator

            # Using nomic-mlx model with explicit mlx device
            generator = EmbeddingGenerator(model_config="nomic-mlx", device="mlx")

            assert generator._mlx_generator is not None
            assert generator.device == "mlx"

    def test_generator_mlx_embed_code_delegates(self) -> None:
        """Test embed_code routes to MLX generator."""
        mock_mlx_generator = MagicMock()
        mock_mlx_generator.embed_code.return_value = [0.1] * 768
        mock_mlx_class = MagicMock(return_value=mock_mlx_generator)

        with patch(
            "codesearch.embeddings.mlx.MLXEmbeddingGenerator", mock_mlx_class
        ):
            from codesearch.embeddings.generator import EmbeddingGenerator

            generator = EmbeddingGenerator(model_config="nomic-mlx")
            result = generator.embed_code("def foo(): pass")

            mock_mlx_generator.embed_code.assert_called_once_with("def foo(): pass")
            assert result == [0.1] * 768

    def test_generator_mlx_embed_batch_delegates(self) -> None:
        """Test embed_batch routes to MLX generator."""
        mock_mlx_generator = MagicMock()
        mock_mlx_generator.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]
        mock_mlx_class = MagicMock(return_value=mock_mlx_generator)

        with patch(
            "codesearch.embeddings.mlx.MLXEmbeddingGenerator", mock_mlx_class
        ):
            from codesearch.embeddings.generator import EmbeddingGenerator

            generator = EmbeddingGenerator(model_config="nomic-mlx")
            result = generator.embed_batch(["code1", "code2"])

            mock_mlx_generator.embed_batch.assert_called_once_with(["code1", "code2"])
            assert len(result) == 2

    def test_generator_mlx_get_model_info_delegates(self) -> None:
        """Test get_model_info routes to MLX generator."""
        mock_mlx_generator = MagicMock()
        mock_mlx_generator.get_model_info.return_value = {
            "name": "nomic-mlx",
            "device": "mlx",
            "dimensions": 768,
        }
        mock_mlx_class = MagicMock(return_value=mock_mlx_generator)

        with patch(
            "codesearch.embeddings.mlx.MLXEmbeddingGenerator", mock_mlx_class
        ):
            from codesearch.embeddings.generator import EmbeddingGenerator

            generator = EmbeddingGenerator(model_config="nomic-mlx")
            info = generator.get_model_info()

            mock_mlx_generator.get_model_info.assert_called_once()
            assert info["name"] == "nomic-mlx"
            assert info["device"] == "mlx"

    def test_generator_mlx_embed_batch_empty_list(self) -> None:
        """Test embed_batch with empty list doesn't delegate."""
        mock_mlx_generator = MagicMock()
        mock_mlx_class = MagicMock(return_value=mock_mlx_generator)

        with patch(
            "codesearch.embeddings.mlx.MLXEmbeddingGenerator", mock_mlx_class
        ):
            from codesearch.embeddings.generator import EmbeddingGenerator

            generator = EmbeddingGenerator(model_config="nomic-mlx")
            result = generator.embed_batch([])

            # Empty list should return early, not delegate
            mock_mlx_generator.embed_batch.assert_not_called()
            assert result == []

    def test_generator_all_mlx_models_route_correctly(self) -> None:
        """Test that all MLX models route to MLXEmbeddingGenerator."""
        mlx_models = ["nomic-mlx", "bge-m3-mlx", "bge-large-mlx", "bge-small-mlx"]

        for model_name in mlx_models:
            mock_mlx_generator = MagicMock()
            mock_mlx_class = MagicMock(return_value=mock_mlx_generator)

            with patch(
                "codesearch.embeddings.mlx.MLXEmbeddingGenerator", mock_mlx_class
            ):
                from codesearch.embeddings.generator import EmbeddingGenerator

                generator = EmbeddingGenerator(model_config=model_name)

                assert generator._mlx_generator is not None, f"{model_name} should route to MLX"
                assert generator.device == "mlx", f"{model_name} should have mlx device"


class TestNonMLXModelsUnchanged:
    """Tests to verify non-MLX models still work unchanged."""

    def test_non_mlx_model_does_not_route_to_mlx(self) -> None:
        """Test that non-MLX models don't route to MLXEmbeddingGenerator."""
        # Mock transformers to avoid loading actual models
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        with (
            patch("codesearch.embeddings.mlx.MLXEmbeddingGenerator") as mock_mlx_class,
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModel.from_pretrained", return_value=mock_model),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            from codesearch.embeddings.generator import EmbeddingGenerator

            generator = EmbeddingGenerator(model_config="codebert")

            assert generator._mlx_generator is None
            assert generator._api_generator is None
            mock_mlx_class.assert_not_called()

    def test_api_model_does_not_route_to_mlx(self) -> None:
        """Test that API models don't route to MLXEmbeddingGenerator."""
        mock_voyage = MagicMock()

        with (
            patch("codesearch.embeddings.mlx.MLXEmbeddingGenerator") as mock_mlx_class,
            patch("codesearch.embeddings.voyage.VoyageEmbeddingGenerator", return_value=mock_voyage),
            patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}),
        ):
            from codesearch.embeddings.generator import EmbeddingGenerator

            generator = EmbeddingGenerator(model_config="voyage-code-3")

            assert generator._mlx_generator is None
            assert generator._api_generator is not None
            mock_mlx_class.assert_not_called()


class TestMLXGeneratorInitialization:
    """Tests for MLX generator initialization parameters."""

    def test_mlx_generator_receives_correct_params(self) -> None:
        """Test MLXEmbeddingGenerator is initialized with correct parameters."""
        mock_mlx_generator = MagicMock()
        mock_mlx_class = MagicMock(return_value=mock_mlx_generator)

        with patch(
            "codesearch.embeddings.mlx.MLXEmbeddingGenerator", mock_mlx_class
        ):
            from codesearch.embeddings.generator import EmbeddingGenerator

            EmbeddingGenerator(model_config="bge-small-mlx")

            # Check MLXEmbeddingGenerator was called with correct params
            mock_mlx_class.assert_called_once_with(
                model_name="bge-small-mlx",
                dimensions=384,  # bge-small-mlx has 384 dimensions
            )
