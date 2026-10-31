import pandas as pd
import pytest

from app.service import MAX_BATCH_ROWS, InputError


def test_model_loads(service):
    assert len(service.features) == 30
    assert 0 < service.threshold < 1


def test_samples_are_scored_as_labelled_or_missed_once(service, samples):
    """The demo samples are 2 caught frauds, 1 missed fraud and 3 legit transactions."""
    outcomes = {"caught": 0, "missed": 0, "legit_ok": 0}
    for s in samples:
        r = service.predict_one(s["features"])
        assert 0 <= r["fraud_score"] <= 1
        if s["actual"] == "fraud":
            outcomes["caught" if r["is_fraud"] else "missed"] += 1
        else:
            assert not r["is_fraud"]
            outcomes["legit_ok"] += 1
    assert outcomes["caught"] >= 2 and outcomes["legit_ok"] >= 3


def test_prediction_shape(service, samples):
    r = service.predict_one(samples[0]["features"])
    assert set(r) == {"fraud_score", "is_fraud", "risk_level", "threshold", "signals"}
    assert r["risk_level"] in {"low", "medium", "high"}
    assert (r["risk_level"] == "high") == r["is_fraud"]


def test_missing_feature_is_rejected(service, samples):
    f = dict(samples[0]["features"])
    f.pop("V14")
    with pytest.raises(InputError, match="V14"):
        service.predict_one(f)


@pytest.mark.parametrize("field,value,msg", [("Amount", -5, "negative"), ("Hour", 24, "Hour"), ("V1", float("nan"), "non-numeric")])
def test_bad_values_are_rejected(service, samples, field, value, msg):
    f = dict(samples[0]["features"])
    f[field] = value
    with pytest.raises(InputError, match=msg):
        service.predict_one(f)


def test_batch_scores_every_row_and_uses_labels(service, samples):
    df = pd.DataFrame([{**s["features"], "Class": int(s["actual"] == "fraud")} for s in samples])
    out = service.predict_batch(df)
    assert out["summary"]["rows"] == len(samples)
    assert out["summary"]["actual_fraud"] == sum(s["actual"] == "fraud" for s in samples)
    assert len(out["results"]) == len(samples)


def test_batch_accepts_time_instead_of_hour(service, samples):
    row = {k: v for k, v in samples[0]["features"].items() if k != "Hour"}
    row["Time"] = 3600 * 5 + 10
    out = service.predict_batch(pd.DataFrame([row]))
    assert out["summary"]["rows"] == 1


def test_batch_limits_and_empty_input(service, samples):
    with pytest.raises(InputError):
        service.predict_batch(pd.DataFrame())
    big = pd.DataFrame([samples[0]["features"]] * (MAX_BATCH_ROWS + 1))
    with pytest.raises(InputError, match="Too many rows"):
        service.predict_batch(big)
