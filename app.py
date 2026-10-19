
"""FastAPI wrapper for the saved harvest decision-support models."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from predict import predict_record
from src.pipeline import ROOT
import json

app = FastAPI(title="Musanze Harvest Intelligence Service", version="1.0.0",
              description="Classroom prototype using synthetic data; human review is required.")

class FarmRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rainfall_mm: float = Field(ge=600, le=1400)
    avg_temp_c: float = Field(ge=12, le=26)
    soil_moisture_pct: float = Field(ge=20, le=80)
    fertilizer_kg: float = Field(ge=0, le=250)
    field_size_ha: float = Field(ge=0.2, le=4.0)
    season: Literal["A","B"]
    cooperative_support: int = Field(ge=0, le=1)

@app.get("/health")
def health():
    return {"status":"ok", "service":"Musanze Harvest Intelligence Service"}

@app.get("/model-info")
def model_info():
    summary_path = ROOT/"artifacts"/"run_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=503, detail="Models are not trained. Run run_all.py first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"verification_code":summary["verification_code"],
            "student_id":summary["student_id"], "selected_models":{
                "regression":summary["result"]["selected_reg"],
                "classification":summary["result"]["selected_clf"],
                "clustering_k":summary["result"]["selected_k"]},
            "dataset":"synthetic classroom dataset"}

@app.post("/predict")
def predict(request: FarmRecord):
    try:
        return predict_record(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
