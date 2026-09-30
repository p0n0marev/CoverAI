from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.config import settings
from app.generator import ImageGenerator


class TestImageGenerator:

    @patch("app.generator.InferenceClient")
    def test_generate_calls_text_to_image(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.text_to_image.return_value = Image.new("RGB", (64, 64))

        result = ImageGenerator().generate("a red square", 128, 256, 7)

        mock_client.text_to_image.assert_called_once_with(
            "a red square",
            model="test-model",
            width=128,
            height=256,
            seed=7,
        )
        assert result.startswith(b"\x89PNG")

    @patch("app.generator.InferenceClient")
    def test_init_creates_client_with_settings(self, mock_client_cls):
        ImageGenerator()

        mock_client_cls.assert_called_once_with(provider="auto", api_key="test-token")

    @patch("app.generator.InferenceClient")
    def test_init_requires_token(self, mock_client_cls, monkeypatch):
        monkeypatch.setattr(settings, "hf_token", None)

        with pytest.raises(RuntimeError):
            ImageGenerator()

        mock_client_cls.assert_not_called()

    @patch("app.generator.apply_proxy")
    @patch("app.generator.InferenceClient")
    def test_init_applies_proxy_before_client(self, mock_client_cls, mock_apply):
        ImageGenerator()

        mock_apply.assert_called_once()
        mock_client_cls.assert_called_once()
