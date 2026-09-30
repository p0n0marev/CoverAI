import uuid
from unittest.mock import MagicMock, patch

import pytest
import redis
from PIL import Image
from rq.exceptions import NoSuchJobError

from app.config import settings
from app.jobs import generate_image_job


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_queue_returns_job_id(client):
    job = MagicMock()
    job.id = "job-123"
    with patch("app.main.queue.enqueue", return_value=job) as enqueue:
        response = client.post("/queue", json={"prompt": "a red square"})

    assert response.status_code == 200
    assert response.json() == {"job_id": "job-123"}
    enqueue.assert_called_once_with(
        generate_image_job,
        "a red square",
        None,
        None,
        None,
        result_ttl=settings.result_ttl,
        failure_ttl=settings.result_ttl,
    )


def test_queue_passes_size_and_seed(client):
    job = MagicMock()
    job.id = "job-456"
    with patch("app.main.queue.enqueue", return_value=job) as enqueue:
        response = client.post(
            "/queue",
            json={"prompt": "cover", "width": 512, "height": 768, "seed": 7},
        )

    assert response.status_code == 200
    enqueue.assert_called_once_with(
        generate_image_job,
        "cover",
        512,
        768,
        7,
        result_ttl=settings.result_ttl,
        failure_ttl=settings.result_ttl,
    )


def test_queue_missing_prompt(client):
    response = client.post("/queue", json={})
    assert response.status_code == 422


def test_queue_empty_prompt(client):
    response = client.post("/queue", json={"prompt": ""})
    assert response.status_code == 422


def test_queue_width_out_of_range(client):
    response = client.post("/queue", json={"prompt": "cover", "width": 64})
    assert response.status_code == 422


def test_queue_wrong_content_type(client):
    response = client.post("/queue", data="not json")
    assert response.status_code == 422


def test_queue_empty_body(client):
    response = client.post("/queue")
    assert response.status_code == 422


def test_queue_extra_fields_ignored(client):
    job = MagicMock()
    job.id = "job-extra"
    with patch("app.main.queue.enqueue", return_value=job):
        response = client.post("/queue", json={"prompt": "test", "extra": "ignored"})
    assert response.status_code == 200


def test_queue_rate_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    with (
        patch("app.main.redis.incr", return_value=2),
        patch("app.main.queue.enqueue") as enqueue,
    ):
        response = client.post("/queue", json={"prompt": "cover"})

    assert response.status_code == 429
    enqueue.assert_not_called()


def test_auth_required_except_health(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret")
    assert client.get("/health").status_code == 200
    assert client.post("/queue", json={"prompt": "cover"}).status_code == 401
    assert client.get("/result/missing").status_code == 401

    rejected = client.post(
        "/queue",
        json={"prompt": "cover"},
        headers={"X-API-Key": "nope"},
    )
    assert rejected.status_code == 401

    job = MagicMock()
    job.id = "job-auth"
    with patch("app.main.queue.enqueue", return_value=job):
        accepted = client.post(
            "/queue",
            json={"prompt": "cover"},
            headers={"X-API-Key": "secret"},
        )
    assert accepted.status_code == 200


def _job(**overrides):
    job = MagicMock()
    job.is_finished = False
    job.is_failed = False
    job.get_status.return_value = "queued"
    job.result = None
    for key, value in overrides.items():
        setattr(job, key, value)
    return job


def test_result_not_found(client):
    with patch("app.main.Job.fetch", side_effect=NoSuchJobError("missing")):
        response = client.get("/result/missing")
    assert response.status_code == 404


def test_result_redis_error_is_not_a_404(client):
    with (
        patch("app.main.Job.fetch", side_effect=redis.ConnectionError("down")),
        pytest.raises(redis.ConnectionError),
    ):
        client.get("/result/missing")


def test_result_pending(client):
    job_id = "job-pending"
    with patch("app.main.Job.fetch", return_value=_job()):
        response = client.get(f"/result/{job_id}")
    assert response.status_code == 200
    assert response.json() == {"id": job_id, "status": "queued"}


def test_result_failed(client):
    with patch("app.main.Job.fetch", return_value=_job(is_failed=True)):
        response = client.get("/result/job-failed")
    assert response.status_code == 500
    assert response.json()["detail"] == "Image generation failed"


def test_result_finished_png(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    image_id = str(uuid.uuid4())
    Image.new("RGB", (8, 8), color="red").save(tmp_path / f"{image_id}.png")
    with patch("app.main.Job.fetch", return_value=_job(is_finished=True, result=image_id)):
        response = client.get(f"/result/{image_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    image = Image.open(__import__("io").BytesIO(response.content))
    assert image.format == "PNG"
    assert image.size == (8, 8)


def test_result_finished_missing_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    image_id = str(uuid.uuid4())
    with patch("app.main.Job.fetch", return_value=_job(is_finished=True, result=image_id)):
        response = client.get(f"/result/{image_id}")
    assert response.status_code == 500
    assert response.json()["detail"] == "Generated file not found"


def test_result_rejects_non_uuid_result(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    outside = tmp_path.parent / "secret.png"
    outside.write_bytes(b"nope")
    with patch("app.main.Job.fetch", return_value=_job(is_finished=True, result="../../secret")):
        response = client.get("/result/job-evil")
    assert response.status_code == 500
    outside.unlink()
