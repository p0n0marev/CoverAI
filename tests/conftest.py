import os

os.environ["HF_TOKEN"] = "test-token"
os.environ["HF_MODEL"] = "test-model"
os.environ["HF_PROVIDER"] = "auto"
os.environ["DEBUG"] = "false"
os.environ["VERSION"] = "test"
os.environ["API_KEY"] = ""
os.environ["RATE_LIMIT_PER_MINUTE"] = "0"
os.environ["RESULT_TTL"] = "86400"

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)
