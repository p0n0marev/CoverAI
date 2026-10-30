import io

from PIL import Image

from app.config import settings
from app.generator import COVER_SIZE, ImageGenerator


def test_generate_png(monkeypatch):
    seen = []

    def render(self, prompt, seed, steps, guidance, size):
        seen.append((prompt, seed, steps, guidance, size))
        return Image.new("RGB", (size, size), "blue")

    monkeypatch.setattr(settings, "steps", 8)
    monkeypatch.setattr(settings, "guidance", 7.5)
    monkeypatch.setattr(settings, "gen_size", 300)
    monkeypatch.setattr(ImageGenerator, "_render", render)

    result = ImageGenerator().generate("cover", 3)
    ImageGenerator().generate("cover", None, 4, 6.5)

    assert seen == [
        ("cover", 3, 8, 7.5, 296),
        ("cover", None, 4, 6.5, 296),
    ]
    with Image.open(io.BytesIO(result)) as image:
        assert image.size == (COVER_SIZE, COVER_SIZE)
