import logging
import time
import uuid
from pathlib import Path

from redis import Redis
from rq import Queue

from app.config import settings
from app.generator import ImageGenerator

logger = logging.getLogger(__name__)

redis = Redis.from_url(settings.redis_url)
queue = Queue('images', connection=redis, default_timeout=1800)


def cleanup_old_images() -> None:
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


def generate_image_job(prompt, seed=None, steps=None, guidance=None):
    image_id = str(uuid.uuid4())
    logger.info("Job started image_id=%s prompt_len=%d steps=%s guidance=%s", image_id, len(prompt), steps, guidance)
    logger.debug("Job prompt image_id=%s prompt=%s", image_id, prompt[:100])
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(settings.output_dir) / f'{image_id}.png'
    path.write_bytes(ImageGenerator().generate(prompt, seed, steps, guidance))
    cleanup_old_images()
    logger.info("Job finished image_id=%s path=%s", image_id, path)
    return image_id
