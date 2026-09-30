import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    """Read an env var. A missing or blank value falls back to default."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Settings:
    version = _env('VERSION', '0.1.0')
    hf_token = _env('HF_TOKEN')
    hf_provider = _env('HF_PROVIDER', 'fal-ai')
    hf_model = _env('HF_MODEL', 'black-forest-labs/FLUX.1-schnell')
    redis_url = _env('REDIS_URL', 'redis://redis:6379/0')
    output_dir = _env('OUTPUT_DIR', 'data/images')
    api_key = _env('API_KEY')
    debug = (_env('DEBUG', 'false') or '').lower() in ('1', 'true', 'yes')
    proxy = _env('PROXY')
    rate_limit_per_minute = _env_int('RATE_LIMIT_PER_MINUTE', 60)
    result_ttl = _env_int('RESULT_TTL', 86400)


settings = Settings()


def apply_proxy() -> None:
    """Send outbound HTTP through PROXY. Call this in the worker, before the HF client is created."""
    if not settings.proxy:
        return
    os.environ.setdefault('HTTP_PROXY', settings.proxy)
    os.environ.setdefault('HTTPS_PROXY', settings.proxy)
