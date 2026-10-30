from app.config import _env, _env_float, _env_int


def test_blank_and_invalid_env_use_default(monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", "   ")
    monkeypatch.setenv("STEPS", "nope")
    monkeypatch.setenv("GUIDANCE", "nope")
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert _env("OUTPUT_DIR", "data/images") == "data/images"
    assert _env("REDIS_URL", "redis://redis:6379/0") == "redis://redis:6379/0"
    assert _env_int("STEPS", 8) == 8
    assert _env_float("GUIDANCE", 7.5) == 7.5
