import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

os.environ.setdefault("HF_TOKEN", "test-token")
os.environ.setdefault("HF_MODEL", "test-model")
os.environ.setdefault("HF_PROVIDER", "auto")
os.environ.setdefault("DEBUG", "false")


@pytest.fixture
def mock_image():
    return Image.new("RGB", (64, 64), color="red")


@pytest.fixture
def client():
    with patch("app.main.image_generator") as mock_gen:
        mock_gen.generate.return_value = Image.new("RGB", (64, 64), color="red")
        from fastapi.testclient import TestClient
        from app.main import app

        yield TestClient(app)
