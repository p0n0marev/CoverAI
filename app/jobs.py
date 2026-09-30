import io
import logging
import time
import uuid
import zlib
from pathlib import Path

from PIL import Image, ImageDraw
from redis import Redis
from rq import Queue

from app.config import settings
from app.generator import ImageGenerator

logger = logging.getLogger(__name__)

redis = Redis.from_url(settings.redis_url)
queue = Queue('images', connection=redis, default_timeout=600)

DEBUG_DEFAULT_SIZE = 512


def debug_image(prompt, width=None, height=None) -> bytes:
    """Local stub instead of calling the model (DEBUG=true)."""
    size = (width or DEBUG_DEFAULT_SIZE, height or DEBUG_DEFAULT_SIZE)
    crc = zlib.crc32(prompt.encode())
    color = (crc >> 16 & 255, crc >> 8 & 255, crc & 255)
    image = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(image)
    draw.text((16, 16), 'DEBUG MOCK', fill='white')
    draw.multiline_text((16, 36), prompt[:200], fill='white')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def cleanup_old_images() -> None:
    """Drop PNGs older than RESULT_TTL so the volume matches how long RQ keeps the job."""
    if settings.result_ttl <= 0:
        return
    directory = Path(settings.output_dir)
    if not directory.is_dir():
        return
    cutoff = time.time() - settings.result_ttl
    for path in directory.glob('*.png'):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                logger.info("Removed expired image %s", path.name)
        except OSError:
            logger.warning("Could not remove expired image %s", path.name)


def generate_image_job(prompt, width=None, height=None, seed=None):
    image_id = str(uuid.uuid4())
    logger.info("Job started image_id=%s prompt_len=%d", image_id, len(prompt))
    logger.debug("Job prompt image_id=%s prompt=%s", image_id, prompt[:100])
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(settings.output_dir) / f'{image_id}.png'
    if settings.debug:
        logger.debug("DEBUG=true, using mock image instead of model for image_id=%s", image_id)
        content = debug_image(prompt, width, height)
    else:
        content = ImageGenerator().generate(prompt, width, height, seed)
    path.write_bytes(content)
    cleanup_old_images()
    logger.info("Job finished image_id=%s path=%s", image_id, path)
    return image_id
