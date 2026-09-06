from unittest.mock import patch, MagicMock
from PIL import Image


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_returns_png(client):
    response = client.post("/generate", json={"prompt": "a red square"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_generate_returns_valid_image(client):
    response = client.post("/generate", json={"prompt": "a red square"})
    img = Image.open(__import__("io").BytesIO(response.content))
    assert img.format == "PNG"
    assert img.size == (64, 64)


def test_generate_calls_image_generator(client):
    with patch("app.main.image_generator") as mock_gen:
        mock_gen.generate.return_value = Image.new("RGB", (32, 32), color="blue")
        response = client.post("/generate", json={"prompt": "test prompt"})
        mock_gen.generate.assert_called_once_with("test prompt")


def test_generate_missing_prompt(client):
    response = client.post("/generate", json={})
    assert response.status_code == 422


def test_generate_wrong_content_type(client):
    response = client.post("/generate", data="not json")
    assert response.status_code == 422


def test_generate_empty_body(client):
    response = client.post("/generate")
    assert response.status_code == 422


def test_generate_extra_fields_ignored(client):
    response = client.post("/generate", json={"prompt": "test", "extra": "ignored"})
    assert response.status_code == 200
