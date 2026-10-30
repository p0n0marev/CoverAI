import gc
import io
import logging

from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

COVER_SIZE = 256


def _trim_memory() -> None:
    import ctypes

    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)


def _torch():
    import os

    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    import torch

    torch.set_num_threads(2)
    return torch


def _generation_size() -> int:
    size = max(128, min(512, settings.gen_size))
    return size - size % 8


class ImageGenerator:
    def generate(self, prompt: str, seed=None, steps=None, guidance=None) -> bytes:
        steps = settings.steps if steps is None else int(steps)
        guidance = settings.guidance if guidance is None else float(guidance)
        size = _generation_size()
        logger.info(
            "Generating cover steps=%s guidance=%s gen_size=%s out=%s seed=%s",
            steps,
            guidance,
            size,
            COVER_SIZE,
            seed,
        )
        image = self._render(prompt, seed, steps, guidance, size)
        if image.size != (COVER_SIZE, COVER_SIZE):
            image = image.resize((COVER_SIZE, COVER_SIZE), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        result = buffer.getvalue()
        logger.info("Cover generated size=%d bytes", len(result))
        return result

    def _render(self, prompt: str, seed, steps: int, guidance: float, size: int) -> Image.Image:
        torch = _torch()
        model = settings.hf_model
        cond, uncond = self._encode_prompt(torch, model, prompt)
        latents = self._denoise(torch, model, cond, uncond, size, seed, steps, guidance)
        pixels = self._decode(torch, model, latents)
        return Image.fromarray(pixels)

    def _encode_prompt(self, torch, model: str, prompt: str):
        from transformers import CLIPTextModel, CLIPTokenizer

        logger.info("Loading text encoder")
        tokenizer = CLIPTokenizer.from_pretrained(model, subfolder='tokenizer')
        text_encoder = CLIPTextModel.from_pretrained(
            model,
            subfolder='text_encoder',
            torch_dtype=torch.float32,
            variant='fp16',
            low_cpu_mem_usage=True,
        )
        text_encoder.eval()
        tokens = tokenizer(
            [prompt, ''],
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        with torch.inference_mode():
            embeds = text_encoder(tokens.input_ids)[0]
        cond, uncond = embeds[0:1], embeds[1:2]
        del text_encoder, tokenizer, tokens, embeds
        _trim_memory()
        return cond, uncond

    def _denoise(self, torch, model: str, cond, uncond, size: int, seed, steps: int, guidance: float):
        from diffusers import EulerDiscreteScheduler, UNet2DConditionModel

        logger.info("Loading UNet")
        unet = UNet2DConditionModel.from_pretrained(
            model,
            subfolder='unet',
            torch_dtype=torch.float32,
            variant='fp16',
            low_cpu_mem_usage=True,
        )
        unet.eval()
        unet.set_attention_slice('auto')
        scheduler = EulerDiscreteScheduler.from_pretrained(model, subfolder='scheduler')
        scheduler.set_timesteps(steps)

        generator = None
        if seed is not None:
            generator = torch.Generator(device='cpu').manual_seed(int(seed))
        latents = torch.randn(
            (1, unet.config.in_channels, size // 8, size // 8),
            generator=generator,
            dtype=torch.float32,
        )
        latents = latents * scheduler.init_noise_sigma
        cond = cond.to(dtype=torch.float32)
        uncond = uncond.to(dtype=torch.float32)

        for timestep in scheduler.timesteps:
            latent_input = scheduler.scale_model_input(latents, timestep)
            with torch.inference_mode():
                noise = unet(latent_input, timestep, encoder_hidden_states=cond).sample
                if guidance != 1:
                    noise_uncond = unet(latent_input, timestep, encoder_hidden_states=uncond).sample
                    noise = noise_uncond + guidance * (noise - noise_uncond)
            latents = scheduler.step(noise, timestep, latents).prev_sample

        del unet, scheduler, cond, uncond
        _trim_memory()
        return latents

    def _decode(self, torch, model: str, latents):
        import numpy as np
        from diffusers import AutoencoderKL

        logger.info("Loading VAE")
        vae = AutoencoderKL.from_pretrained(
            model,
            subfolder='vae',
            torch_dtype=torch.float32,
            variant='fp16',
            low_cpu_mem_usage=True,
        )
        vae.eval()
        vae.enable_slicing()
        vae.enable_tiling()
        latents = latents.to(dtype=torch.float32) / vae.config.scaling_factor
        with torch.inference_mode():
            image = vae.decode(latents).sample
        del vae, latents
        _trim_memory()
        image = (image / 2 + 0.5).clamp(0, 1)
        array = image.permute(0, 2, 3, 1).numpy()[0]
        return (array * 255).round().astype(np.uint8)
