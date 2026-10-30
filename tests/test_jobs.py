import os
import time
import uuid

from app.config import settings
from app.generator import ImageGenerator
from app.jobs import generate_image_job


def test_job_writes_image_and_drops_expired_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ImageGenerator, "generate", lambda self, *args, **kwargs: b"png")
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    monkeypatch.setattr(settings, "result_ttl", 60)

    expired = tmp_path / f"{uuid.uuid4()}.png"
    expired.write_bytes(b"old")
    os.utime(expired, (time.time() - 120, time.time() - 120))

    image_id = generate_image_job("hello cover")

    assert (tmp_path / f"{image_id}.png").read_bytes() == b"png"
    assert not expired.exists()
