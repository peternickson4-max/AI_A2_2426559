
import json, subprocess, sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from predict import predict_record, validate_record
from app import app
client=TestClient(app)
VALID=json.loads((ROOT/"samples"/"valid.json").read_text())

def test_valid_inference():
    result=predict_record(VALID)
    assert result["estimated_yield_tons"] >= 0.5
    assert result["high_risk_prediction"] in (0,1)
    assert 0 <= result["high_risk_probability"] <= 1
    assert result["verification_code"]=="AI-A2-6559"

def test_missing_field_rejected():
    bad=VALID.copy(); bad.pop("season")
    with pytest.raises(ValueError, match="Missing required fields"):
        validate_record(bad)

def test_wrong_type_rejected():
    bad=VALID.copy(); bad["rainfall_mm"]="wet"
    with pytest.raises(ValueError, match="must be numeric"):
        validate_record(bad)

def test_range_error_rejected():
    bad=VALID.copy(); bad["rainfall_mm"]=1500
    with pytest.raises(ValueError, match="between 600 and 1400"):
        validate_record(bad)

def test_unexpected_field_rejected():
    bad=VALID.copy(); bad["yield_tons"]=99
    with pytest.raises(ValueError, match="Unexpected fields"):
        validate_record(bad)

def test_health_api_response():
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()["status"]=="ok"

def test_predict_api_response():
    response=client.post("/predict",json=VALID)
    assert response.status_code==200
    assert response.json()["verification_code"]=="AI-A2-6559"

def test_api_range_error():
    bad=VALID.copy(); bad["rainfall_mm"]=1500
    response=client.post("/predict",json=bad)
    assert response.status_code==422

def test_reproducibility_and_seed_guard():
    from data.generate_data import generate_dataset
    a=generate_dataset(6559)
    b=generate_dataset(6559)
    assert a.equals(b)
    from src.pipeline import run_all
    with pytest.raises(ValueError, match="Seed must equal"):
        run_all("student@ines.ac.rw", 1234)
