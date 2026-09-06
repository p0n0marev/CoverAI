import os
from warnings import deprecated

from huggingface_hub import InferenceClient

class ImageGenerator:

    def __init__(self):
        self.client = InferenceClient(
            provider=os.getenv('HF_PROVIDER'),
            api_key=os.environ['HF_TOKEN'],
        )

    def generate(self, prompt: str):
        return self.client.text_to_image(
            prompt,
            model=os.getenv('HF_MODEL'),
        )