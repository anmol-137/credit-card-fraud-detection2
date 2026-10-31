"""HTTP-level tests. Requires fastapi, httpx and python-multipart (see requirements-dev.txt)."""
import io

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # the context manager runs the startup (model loading)
        yield c


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "model_loaded": True}


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Fraud Desk" in r.text


def test_model_card_and_samples(client):
    assert client.get("/api/model").json()["deployed_model"] == "Random Forest"
    assert len(client.get("/api/samples").json()) == 6


def test_predict_ok(client, samples):
    r = client.post("/api/predict", json={"features": samples[0]["features"]})
    assert r.status_code == 200
    assert 0 <= r.json()["fraud_score"] <= 1


def test_predict_missing_feature_gives_422(client):
    r = client.post("/api/predict", json={"features": {"Amount": 10, "Hour": 3}})
    assert r.status_code == 422 and "Missing features" in r.json()["detail"]


def test_predict_wrong_type_gives_422(client):
    assert client.post("/api/predict", json={"features": "nope"}).status_code == 422


def test_batch_upload(client, samples):
    cols = list(samples[0]["features"])
    csv = ",".join(cols) + "\n" + "\n".join(",".join(str(s["features"][c]) for c in cols) for s in samples) + "\n"
    r = client.post("/api/predict/batch", files={"file": ("tx.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200 and r.json()["summary"]["rows"] == len(samples)


def test_batch_rejects_non_csv(client):
    r = client.post("/api/predict/batch", files={"file": ("x.csv", io.BytesIO(b"\x00\x01\x02"), "text/csv")})
    assert r.status_code == 422
