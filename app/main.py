import hashlib
import hmac
import logging
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app.config import settings
from app.jobs import generate_image_job, queue, redis

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title='CoverAI', version=settings.version)

API_KEY_NAME = 'X-API-Key'
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def _keys_match(provided: str | None, expected: str) -> bool:
    if not provided:
        return False
    return hmac.compare_digest(provided.encode(), expected.encode())


def auth(api_key: str | None = Security(api_key_header)):
    if settings.api_key and not _keys_match(api_key, settings.api_key):
        raise HTTPException(401, 'Invalid API key')


def enforce_rate_limit(api_key: str | None) -> None:
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    identity = api_key or 'anonymous'
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    bucket = f'ratelimit:{digest}:{int(time.time()) // 60}'
    count = redis.incr(bucket)
    if count == 1:
        redis.expire(bucket, 120)
    if count > limit:
        raise HTTPException(429, 'Rate limit exceeded')


def finished_image_path(image_id: object) -> Path:
    try:
        normalized = str(uuid.UUID(str(image_id)))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(500, 'Generated file not found')
    root = Path(settings.output_dir).resolve()
    path = (root / f'{normalized}.png').resolve()
    if not path.is_relative_to(root):
        raise HTTPException(500, 'Generated file not found')
    return path


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    steps: int | None = Field(None, ge=1, le=50)
    guidance: float | None = Field(None, ge=0, le=20)
    seed: int | None = None


class QueuedResponse(BaseModel):
    job_id: str


JOB_STATUS_SCHEMA = {
    'type': 'object',
    'properties': {
        'id': {'type': 'string'},
        'status': {'type': 'string'},
    },
    'required': ['id', 'status'],
}


@app.get('/health', tags=['health'])
def health() -> HealthResponse:
    return {'status': 'ok'}


@app.post(
    '/queue',
    tags=['images'],
    response_model=QueuedResponse,
    responses={
        401: {'description': 'Unauthorized', 'model': ErrorResponse},
        429: {'description': 'Too many requests', 'model': ErrorResponse},
    },
    dependencies=[Depends(auth)],
)
def enqueue_image(
    request: GenerateRequest,
    api_key: str | None = Security(api_key_header),
):
    enforce_rate_limit(api_key)
    ttl = settings.result_ttl if settings.result_ttl > 0 else -1
    job = queue.enqueue(
        generate_image_job,
        request.prompt,
        request.seed,
        request.steps,
        request.guidance,
        result_ttl=ttl,
        failure_ttl=ttl,
    )
    logger.info("POST /queue job_id=%s prompt_len=%d", job.id, len(request.prompt))
    logger.debug("POST /queue job_id=%s prompt=%s", job.id, request.prompt[:100])
    return {'job_id': job.id}


@app.get(
    '/result/{job_id}',
    tags=['images'],
    responses={
        200: {
            'description': 'PNG or status',
            'content': {
                'image/png': {'schema': {'type': 'string', 'format': 'binary'}},
                'application/json': {'schema': JOB_STATUS_SCHEMA},
            },
        },
        401: {'description': 'Unauthorized', 'model': ErrorResponse},
        404: {'description': 'Not found', 'model': ErrorResponse},
        500: {'description': 'Failed', 'model': ErrorResponse},
    },
    dependencies=[Depends(auth)],
)
def result(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis)
    except NoSuchJobError:
        logger.warning("GET /result/%s - job not found", job_id)
        raise HTTPException(404, 'Job not found')

    if job.is_finished:
        path = finished_image_path(job.result)
        logger.info("GET /result/%s - finished, path=%s exists=%s", job_id, path, path.exists())
        if not path.exists():
            raise HTTPException(500, 'Generated file not found')
        return FileResponse(path, media_type='image/png', filename=path.name)
    if job.is_failed:
        logger.error("GET /result/%s - job failed", job_id)
        raise HTTPException(500, 'Image generation failed')

    return {'id': job_id, 'status': job.get_status()}
