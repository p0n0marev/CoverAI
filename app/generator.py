import io
import logging

from huggingface_hub import InferenceClient

from app.config import apply_proxy, settings

logger = logging.getLogger(__name__)


class ImageGenerator:
    def __init__(self):
        if not settings.hf_token:
            raise RuntimeError('HF_TOKEN is required')
        apply_proxy()
        self.client = InferenceClient(
            provider=settings.hf_provider,
            api_key=settings.hf_token,
        )

    def generate(self, prompt: str, width=None, height=None, seed=None) -> bytes:
        kwargs = {'model': settings.hf_model}
        if width is not None:
            kwargs['width'] = width
        if height is not None:
            kwargs['height'] = height
        if seed is not None:
            kwargs['seed'] = seed

        logger.info(
            "Generating image model=%s width=%s height=%s seed=%s",
            settings.hf_model,
            width,
            height,
            seed,
        )
        image = self.client.text_to_image(prompt, **kwargs)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        result = buffer.getvalue()
        logger.info("Image generated size=%d bytes", len(result))
        return result
