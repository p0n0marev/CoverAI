import io
import uuid
from unittest.mock import MagicMock, patch

from PIL import Image
from rq.exceptions import NoSuchJobError

from app.config import settings
from app.jobs import generate_image_job


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_queue(client):
    job = MagicMock()
    job.id = "job-1"
    with patch("app.main.queue.enqueue", return_value=job) as enqueue:
        response = client.post(
            "/queue",
            json={"prompt": "cover", "seed": 7, "steps": 4, "guidance": 6.5},
        )

    assert response.json() == {"job_id": "job-1"}
    enqueue.assert_called_once_with(
        generate_image_job,
        "cover",
        7,
        4,
        6.5,
        result_ttl=settings.result_ttl,
        failure_ttl=settings.result_ttl,
    )


def test_queue_rejects_invalid_body(client):
    assert client.post("/queue", json={"prompt": ""}).status_code == 422
    assert client.post("/queue", json={"prompt": "cover", "steps": 0}).status_code == 422


def test_queue_rate_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    with (
        patch("app.main.redis.incr", return_value=2),
        patch("app.main.queue.enqueue") as enqueue,
    ):
        response = client.post("/queue", json={"prompt": "cover"})

    assert response.status_code == 429
    enqueue.assert_not_called()


def test_auth(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret")
    assert client.get("/health").status_code == 200
    assert client.post("/queue", json={"prompt": "cover"}).status_code == 401
    assert client.post(
        "/queue",
        json={"prompt": "cover"},
        headers={"X-API-Key": "nope"},
    ).status_code == 401

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


def test_result_pending(client):
    with patch("app.main.Job.fetch", return_value=_job()):
        response = client.get("/result/job-pending")
    assert response.json() == {"id": "job-pending", "status": "queued"}


def test_result_errors(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    image_id = str(uuid.uuid4())

    with patch("app.main.Job.fetch", side_effect=NoSuchJobError("missing")):
        assert client.get("/result/missing").status_code == 404

    with patch("app.main.Job.fetch", return_value=_job(is_failed=True)):
        assert client.get("/result/job-failed").status_code == 500

    with patch("app.main.Job.fetch", return_value=_job(is_finished=True, result=image_id)):
        assert client.get(f"/result/{image_id}").status_code == 500

    with patch("app.main.Job.fetch", return_value=_job(is_finished=True, result="../secret")):
        assert client.get("/result/job-evil").status_code == 500


def test_result_png(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    image_id = str(uuid.uuid4())
    Image.new("RGB", (8, 8), color="red").save(tmp_path / f"{image_id}.png")
    with patch("app.main.Job.fetch", return_value=_job(is_finished=True, result=image_id)):
        response = client.get(f"/result/{image_id}")

    assert response.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.size == (8, 8)
