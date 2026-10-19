# Musanze Harvest Intelligence Service

**Full name:** Shyaka Peter Guy  
**Registration number:** 24/26559  
**Institutional username:** 2426559@ines.ac.rw  
**Verification code:** AI-A2-6559

## Problem and intended use
This individual project builds a reproducible classroom prototype for a Musanze potato cooperative. It estimates harvest quantity, classifies high production risk, and groups farms with similar input conditions. **All data are synthetic**, not collected from real farms.

## Environment
- Python 3.11 recommended (tested environment should be recorded by the student after running locally).
- Compatible Windows, macOS, or Linux terminal.
- Dependencies are listed in `requirements.txt`.

## Setup and reproduce
From the extracted `AI_A2_2426559` folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd:
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python run_all.py --student-id 2426559@ines.ac.rw --seed 6559
pytest -q
python predict.py --json samples/valid.json
python predict.py --json samples/invalid.json
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the API. The seed is fixed to the last four registration digits. A different seed is intentionally rejected.

## API contract
- `GET /health`: service status.
- `GET /model-info`: verification code, selected models, and synthetic-data notice.
- `POST /predict`: JSON fields `rainfall_mm`, `avg_temp_c`, `soil_moisture_pct`, `fertilizer_kg`, `field_size_ha`, `season` (`A` or `B`), and `cooperative_support` (`0` or `1`).
- Invalid fields/ranges return a 4xx response (FastAPI/Pydantic validation normally returns 422).

Example request:
```json
{
  "rainfall_mm": 950.0, "avg_temp_c": 18.5,
  "soil_moisture_pct": 55.0, "fertilizer_kg": 120.0,
  "field_size_ha": 1.4, "season": "A", "cooperative_support": 1
}
```

## Design and leakage controls
`data/generate_data.py` creates the individual deterministic CSV. `src/pipeline.py` validates the schema, records missing values and duplicates, removes exact duplicates, splits before fitting preprocessing, and uses scikit-learn pipelines. Classification uses only the seven original input features; neither yield nor high-risk labels are features. Clustering also excludes targets. The selected regression model is based on lowest held-out MAE; the classifier is selected by highest held-out F1. K is selected using silhouette score over K=2…6.

## Artifact guide
- `artifacts/data_fingerprint.txt`, `data_validation.json`: reproducibility and data-quality evidence.
- `artifacts/regression_metrics.json`, `classification_metrics.json`, `clustering_report.json`: evaluation evidence.
- `artifacts/test_predictions.csv`: held-out values, predictions, probabilities, residuals and clusters.
- `artifacts/plots/`: residual, confusion-matrix and cluster visualizations.
- `models/`: saved full inference pipelines.
- `docs/AI_A2_2426559_REPORT.pdf`: technical report.
- `evidence/`: test log and declaration PDFs plus AI-use disclosure.

## Limitations and responsible use
This model is for classroom demonstration only. It is trained on formula-generated synthetic data, so its metrics do not establish real-world accuracy. It may encode unrealistic assumptions and must not be used to make decisions about loans, insurance, public services, or individual farmer eligibility. Predictions require agronomist review. Before deployment, collect consented, representative, quality-checked farm data, consult agronomists, assess subgroup errors, and monitor drift.

## Repository and commit
Repository URL: **not created yet** (create your own repository if required by the instructor).  
Final commit hash: **not created yet** (record the hash only after you commit the final tested files).

## Evidence still requiring the student's own action
The assignment requires a continuous, unedited narrated demo video with a handwritten identity card, visible terminal clock, live run, API calls, invalid input, tests, report and declaration. This cannot honestly be fabricated from a generated project. Record `evidence/AI_A2_2426559_DEMO.mp4` yourself after running the commands and replace this note with the real video. Also verify the institutional username above before submission.
