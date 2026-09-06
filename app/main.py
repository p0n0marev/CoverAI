import io
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from app.services.image_generator import ImageGenerator


load_dotenv()

DEBUG = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

if DEBUG:
    os.environ.setdefault("HTTP_PROXY", os.environ["PROXY"])
    os.environ.setdefault("HTTPS_PROXY", os.environ["PROXY"])
    logger.debug("DEBUG mode enabled, using proxy %s", os.environ["PROXY"])


app = FastAPI(
    title="CoverAI",
    version="1.0.0",
)

image_generator = ImageGenerator()


class GenerateImageRequest(BaseModel):
    prompt: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate_image(request: GenerateImageRequest):
    logger.debug("Generating image for prompt: %s", request.prompt)

    image = image_generator.generate(request.prompt)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    logger.info("Generated image for prompt %r, size %d bytes", request.prompt, len(buffer.getvalue()))

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
    )