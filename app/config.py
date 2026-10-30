import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
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


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Settings:
    version = _env('VERSION', '0.2.0')
    hf_model = _env('HF_MODEL', 'stable-diffusion-v1-5/stable-diffusion-v1-5')
    steps = _env_int('STEPS', 8)
    guidance = _env_float('GUIDANCE', 7.5)
    gen_size = _env_int('GEN_SIZE', 256)
    redis_url = _env('REDIS_URL', 'redis://redis:6379/0')
    output_dir = _env('OUTPUT_DIR', 'data/images')
    api_key = _env('API_KEY')
    debug = (_env('DEBUG', 'false') or '').lower() in ('1', 'true', 'yes')
    rate_limit_per_minute = _env_int('RATE_LIMIT_PER_MINUTE', 60)
    result_ttl = _env_int('RESULT_TTL', 86400)


settings = Settings()
