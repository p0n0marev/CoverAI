import logging

from rq import Worker

from app.config import settings
from app.jobs import queue

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting worker on queue 'images', model=%s", settings.hf_model)
    Worker([queue], connection=queue.connection).work()


if __name__ == '__main__':
    main()
