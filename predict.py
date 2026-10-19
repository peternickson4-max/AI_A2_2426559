
"""Local JSON prediction command."""
import argparse, json, sys
from pathlib import Path
import joblib
import pandas as pd
from src.pipeline import ROOT, FEATURES, NUMERIC, CATEGORICAL

def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError("Input JSON must be an object.")
    missing = set(FEATURES) - set(record)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    extra = set(record) - set(FEATURES)
    if extra:
        raise ValueError(f"Unexpected fields: {sorted(extra)}")
    ranges = {"rainfall_mm":(600,1400), "avg_temp_c":(12,26),
              "soil_moisture_pct":(20,80), "fertilizer_kg":(0,250),
              "field_size_ha":(0.2,4.0), "cooperative_support":(0,1)}
    for key, (lo, hi) in ranges.items():
        val = record[key]
        if isinstance(val, bool) or not isinstance(val, (int,float)):
            raise ValueError(f"{key} must be numeric.")
        if not lo <= val <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}.")
    if record["cooperative_support"] not in (0,1):
        raise ValueError("cooperative_support must be 0 or 1.")
    if record["season"] not in ("A","B"):
        raise ValueError("season must be A or B.")
    return record

def predict_record(record):
    record = validate_record(record)
    X = pd.DataFrame([{k:record[k] for k in FEATURES}])
    reg = joblib.load(ROOT/"models"/"regression_pipeline.joblib")
    clf = joblib.load(ROOT/"models"/"classification_pipeline.joblib")
    clus = joblib.load(ROOT/"models"/"clustering_pipeline.joblib")
    probability = float(clf.predict_proba(X)[0, list(clf.classes_).index(1)])
    return {"estimated_yield_tons": round(float(reg.predict(X)[0]),4),
            "high_risk_prediction": int(clf.predict(X)[0]),
            "high_risk_probability": round(probability,4),
            "cluster": int(clus["model"].predict(clus["preprocessor"].transform(X))[0]),
            "verification_code": f"AI-A2-{clus['seed']:04d}"}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to one JSON record")
    args=parser.parse_args()
    try:
        record=json.loads(Path(args.json).read_text(encoding="utf-8"))
        print(json.dumps(predict_record(record), indent=2))
    except Exception as exc:
        print(json.dumps({"error":str(exc)}), file=sys.stderr)
        raise SystemExit(2)
if __name__=="__main__": main()
