import hashlib
import hmac
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app.config import settings
from app.jobs import generate_image_job, queue, redis

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

DESCRIPTION = """
Cover generation service: describe the image in text — get a PNG.

### How to use
1. Send `POST /queue` with a prompt — you get back the task's `job_id`.
2. Poll `GET /result/{job_id}`: while generation is in progress you get a status,
   when the task is finished — the ready `image/png` file.

Generation runs asynchronously in a worker through Redis/RQ, so the
`/queue` request does not wait for the model to finish.

### Authorization
If `API_KEY` is set in the configuration, all endpoints except `/health`
require the header `X-API-Key: <API_KEY>`.

`POST /queue` is rate-limited per API key (see `RATE_LIMIT_PER_MINUTE`).
Finished images and job records are kept for `RESULT_TTL` seconds.
"""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not settings.debug and not settings.hf_token:
        logger.warning("HF_TOKEN is empty and DEBUG is false; the worker cannot generate images")
    yield


app = FastAPI(
    title='CoverAI',
    summary='Generate covers from a text description',
    description=DESCRIPTION,
    version=settings.version,
    lifespan=lifespan,
    openapi_tags=[
        {
            'name': 'health',
            'description': 'Service availability check.',
        },
        {
            'name': 'images',
            'description': 'Enqueue generation tasks and fetch their results.',
        },
    ],
)

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
    detail: str = Field(description='Error description')


class HealthResponse(BaseModel):
    status: str = Field(description='`ok` if the service is up')


class GenerateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            'example': {
                'prompt': 'minimalist book cover, futuristic city skyline at dusk',
                'width': 1024,
                'height': 1024,
                'seed': 42,
            }
        }
    )

    prompt: str = Field(
        min_length=1,
        max_length=10000,
        description='Text description of the desired cover',
    )
    width: int | None = Field(
        None,
        ge=256,
        le=2048,
        description='Image width in pixels, if you want to override it',
    )
    height: int | None = Field(
        None,
        ge=256,
        le=2048,
        description='Image height in pixels, if you want to override it',
    )
    seed: int | None = Field(
        None,
        description='Seed for reproducible generation',
    )


class QueuedResponse(BaseModel):
    job_id: str = Field(description='Identifier of the queued task')


JOB_STATUS_SCHEMA = {
    'type': 'object',
    'properties': {
        'id': {'type': 'string', 'description': 'Task identifier'},
        'status': {
            'type': 'string',
            'description': '`queued` or `started`',
        },
    },
    'required': ['id', 'status'],
}

RESPONSES_UNAUTHORIZED = {
    401: {
        'description': 'Invalid or missing API key',
        'model': ErrorResponse,
    },
}


@app.get(
    '/health',
    tags=['health'],
    summary='Liveness probe',
    description='No authorization required. Suitable for Docker/Kubernetes healthchecks.',
    response_description='Service state',
)
def health() -> HealthResponse:
    return {'status': 'ok'}


@app.post(
    '/queue',
    tags=['images'],
    summary='Enqueue a generation task',
    description=(
            'Pushes a task into the `images` queue. Immediately returns a `job_id`, '
            'which can later be used to fetch the image via `GET /result/{job_id}`.'
    ),
    response_model=QueuedResponse,
    response_description='Identifier of the enqueued task',
    responses={
        422: {'description': 'Request body validation error'},
        429: {'description': 'Rate limit exceeded', 'model': ErrorResponse},
        **RESPONSES_UNAUTHORIZED,
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
        request.width,
        request.height,
        request.seed,
        result_ttl=ttl,
        failure_ttl=ttl,
    )
    logger.info("POST /queue job_id=%s prompt_len=%d", job.id, len(request.prompt))
    logger.debug("POST /queue job_id=%s prompt=%s", job.id, request.prompt[:100])
    return {'job_id': job.id}


@app.get(
    '/result/{job_id}',
    tags=['images'],
    summary='Get the task result',
    description=(
            'While the task is queued or running, a JSON status is returned. '
            'Once generation is complete, the ready PNG file is served. '
            'Task timeout: 600 seconds. The result is kept for `RESULT_TTL` seconds.'
    ),
    responses={
        200: {
            'description': 'Ready image (`image/png`) or the current task status',
            'content': {
                'image/png': {'schema': {'type': 'string', 'format': 'binary'}},
                'application/json': {'schema': JOB_STATUS_SCHEMA},
            },
        },
        404: {'description': 'No task with such id', 'model': ErrorResponse},
        500: {'description': 'Generation failed or the file was not found', 'model': ErrorResponse},
        **RESPONSES_UNAUTHORIZED,
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
