import logging
import sys

from rq import Worker

from app.config import apply_proxy, settings
from app.jobs import queue

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    if settings.proxy:
        apply_proxy()
        logger.info("HTTP proxy enabled")
    if not settings.debug and not settings.hf_token:
        logger.error("HF_TOKEN is required when DEBUG is false")
        sys.exit(1)
    logger.info("Starting worker on queue 'images'")
    Worker([queue], connection=queue.connection).work()


if __name__ == '__main__':
    main()
