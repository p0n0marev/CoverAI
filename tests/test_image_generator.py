from unittest.mock import patch, MagicMock
from PIL import Image


class TestImageGenerator:

    @patch("app.services.image_generator.InferenceClient")
    def test_generate_calls_text_to_image(self, mock_client_cls):
        from app.services.image_generator import ImageGenerator

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.text_to_image.return_value = Image.new("RGB", (64, 64))

        gen = ImageGenerator()
        result = gen.generate("a red square")

        mock_client.text_to_image.assert_called_once_with("a red square", model="test-model")
        assert isinstance(result, Image.Image)

    @patch("app.services.image_generator.InferenceClient")
    def test_init_creates_client_with_env_vars(self, mock_client_cls):
        import os
        os.environ["HF_TOKEN"] = "my-token"
        os.environ["HF_PROVIDER"] = "fal-ai"

        from app.services.image_generator import ImageGenerator
        gen = ImageGenerator()

        mock_client_cls.assert_called_once_with(provider="fal-ai", api_key="my-token")

    @patch("app.services.image_generator.InferenceClient")
    def test_generate_returns_pil_image(self, mock_client_cls):
        from app.services.image_generator import ImageGenerator

        expected = Image.new("RGB", (100, 100), color="green")
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.text_to_image.return_value = expected

        gen = ImageGenerator()
        result = gen.generate("describe")

        assert result is expected
