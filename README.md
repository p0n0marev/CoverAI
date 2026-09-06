# CoverAI

AI-powered cover image generator built with FastAPI and Hugging Face Inference API. Send a text prompt, get a PNG image back.

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set your HF_TOKEN

# 4. Run the server
uvicorn app.main:app --port 8011
```

## Configuration

| Variable | Default                            | Description                                 |
|----------|------------------------------------|---------------------------------------------|
| `HF_TOKEN` | *(required)*                       | Hugging Face API token                      |
| `HF_MODEL` | `black-forest-labs/FLUX.1-schnell` | Text-to-image model                         |
| `HF_PROVIDER` | `auto`                             | Hugging Face Inference provider             |
| `DEBUG` | `true`                             | Enable debug mode              |

## API

### `GET /health`

Returns `{"status": "ok"}`.

### `POST /generate`

Generates an image from a text prompt.

```bash
curl -X POST http://localhost:8011/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "minimalist book cover, futuristic city skyline at dusk"}' \
  --output cover.png
```

**Request body:**

```json
{ "prompt": "your image description" }
```

**Response:** `image/png`

## Project Structure

```
app/
├── main.py                  # FastAPI app and endpoints
└── services/
    └── image_generator.py   # Hugging Face InferenceClient wrapper
```

## Tech Stack

- Python 3.14+
- FastAPI
- Hugging Face Hub (`InferenceClient`)
- Pillow
