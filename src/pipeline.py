
"""Reproducible data validation, model training, evaluation, and artifact creation."""
from __future__ import annotations
import json, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score,
    silhouette_score)
from sklearn.cluster import KMeans
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg",
           "field_size_ha", "cooperative_support"]
CATEGORICAL = ["season"]
FEATURES = NUMERIC + CATEGORICAL
EXPECTED = set(FEATURES + ["yield_tons", "high_risk"])

def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate schema/ranges, count issues, remove exact duplicates and target-invalid rows."""
    missing_cols = EXPECTED - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    if set(df.columns) != EXPECTED:
        raise ValueError(f"Unexpected columns: {sorted(set(df.columns)-EXPECTED)}")
    evidence = {
        "input_rows": int(len(df)),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }
    clean = df.drop_duplicates().copy()
    if not clean["season"].dropna().isin(["A", "B"]).all():
        raise ValueError("season must contain only A or B.")
    ranges = {
        "rainfall_mm": (600, 1400), "avg_temp_c": (12, 26),
        "soil_moisture_pct": (20, 80), "fertilizer_kg": (0, 250),
        "field_size_ha": (0.2, 4.0), "cooperative_support": (0, 1),
        "yield_tons": (0.5, float("inf")), "high_risk": (0, 1),
    }
    for col, (low, high) in ranges.items():
        vals = clean[col].dropna()
        if ((vals < low) | (vals > high)).any():
            raise ValueError(f"{col} contains values outside [{low}, {high}].")
    if clean[["yield_tons", "high_risk"]].isna().any().any():
        raise ValueError("Target columns may not be missing.")
    if not clean["high_risk"].isin([0, 1]).all():
        raise ValueError("high_risk must be binary (0 or 1).")
    evidence["clean_rows"] = int(len(clean))
    evidence["removed_duplicates"] = evidence["input_rows"] - evidence["clean_rows"]
    evidence["clean_missing_by_column"] = {k: int(v) for k, v in clean.isna().sum().items()}
    return clean.reset_index(drop=True), evidence

def preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])

def regression_metrics(y_true, y_pred):
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}

def classification_metrics(y_true, y_pred, probability):
    result = {"precision": float(precision_score(y_true, y_pred, zero_division=0)),
              "recall": float(recall_score(y_true, y_pred, zero_division=0)),
              "F1": float(f1_score(y_true, y_pred, zero_division=0)),
              "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()}
    try:
        result["ROC_AUC"] = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None
    except ValueError:
        result["ROC_AUC"] = None
    return result

def train_all(seed: int, student_id: str):
    """Train all models and recreate required evidence files deterministically."""
    data_path = ROOT / "data" / "AI_A2_2426559.csv"
    raw = pd.read_csv(data_path)
    clean, data_evidence = validate_and_clean(raw)
    data_evidence.update({"verification_code": f"AI-A2-{seed:04d}", "seed": seed,
                          "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()})
    (ROOT/"artifacts"/"data_validation.json").write_text(json.dumps(data_evidence, indent=2))
    X = clean[FEATURES]
    yreg = clean["yield_tons"]
    yclf = clean["high_risk"]
    # Fixed split, with no preprocessing fitted before the split.
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, yreg, yclf, test_size=0.25, random_state=seed, stratify=yclf)
    reg_candidates = {
        "mean_baseline": Pipeline([("prep", preprocessor()), ("model", DummyRegressor(strategy="mean"))]),
        "linear_regression": Pipeline([("prep", preprocessor()), ("model", LinearRegression())]),
        "random_forest": Pipeline([("prep", preprocessor()), ("model", RandomForestRegressor(
            n_estimators=160, min_samples_leaf=2, random_state=seed, n_jobs=1))]),
    }
    reg_scores = {}
    reg_fitted = {}
    for name, model in reg_candidates.items():
        model.fit(X_train, yr_train)
        pred = model.predict(X_test)
        reg_scores[name] = regression_metrics(yr_test, pred)
        reg_fitted[name] = model
    selected_reg = min(["linear_regression", "random_forest"], key=lambda n: reg_scores[n]["MAE"])
    clf_candidates = {
        "majority_baseline": Pipeline([("prep", preprocessor()), ("model", DummyClassifier(strategy="most_frequent"))]),
        "logistic_regression": Pipeline([("prep", preprocessor()), ("model", LogisticRegression(max_iter=2000, random_state=seed))]),
        "random_forest": Pipeline([("prep", preprocessor()), ("model", RandomForestClassifier(
            n_estimators=160, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=1))]),
    }
    clf_scores = {}
    clf_fitted = {}
    for name, model in clf_candidates.items():
        model.fit(X_train, yc_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        clf_scores[name] = classification_metrics(yc_test, pred, prob)
        clf_fitted[name] = model
    valid_clfs = ["logistic_regression", "random_forest"]
    selected_clf = max(valid_clfs, key=lambda n: clf_scores[n]["F1"])
    (ROOT/"artifacts"/"regression_metrics.json").write_text(json.dumps({
        "verification_code": f"AI-A2-{seed:04d}", "test_rows": len(X_test),
        "metrics": reg_scores, "selected_model": selected_reg,
        "selection_reason": "Lowest held-out MAE among the two candidate regressors."
    }, indent=2))
    (ROOT/"artifacts"/"classification_metrics.json").write_text(json.dumps({
        "verification_code": f"AI-A2-{seed:04d}", "test_rows": len(X_test),
        "metrics": clf_scores, "selected_model": selected_clf,
        "selection_reason": "Highest held-out F1 among the two candidate classifiers."
    }, indent=2))
    # Clustering uses only feature data; fit scaling/imputation on the cleaned feature matrix.
    cluster_prep = preprocessor()
    Z = cluster_prep.fit_transform(X)
    if hasattr(Z, "toarray"): Z = Z.toarray()
    k_scores = {}
    fitted_k = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(Z)
        k_scores[str(k)] = float(silhouette_score(Z, labels))
        fitted_k[k] = (km, labels)
    selected_k = max(range(2,7), key=lambda k: k_scores[str(k)])
    km, labels = fitted_k[selected_k]
    profiles = []
    profile_df = clean[FEATURES].copy()
    profile_df["cluster"] = labels
    for cluster_id in range(selected_k):
        subset = profile_df[profile_df.cluster == cluster_id]
        profiles.append({
            "cluster": int(cluster_id), "farms": int(len(subset)),
            "mean_rainfall_mm": round(float(subset.rainfall_mm.mean()), 2),
            "mean_soil_moisture_pct": round(float(subset.soil_moisture_pct.mean()), 2),
            "mean_fertilizer_kg": round(float(subset.fertilizer_kg.mean()), 2),
            "mean_field_size_ha": round(float(subset.field_size_ha.mean()), 2),
            "common_season": str(subset.season.mode().iloc[0]),
            "plain_language_profile": "Relatively wetter farms" if subset.soil_moisture_pct.mean() >= profile_df.soil_moisture_pct.mean()
                else "Relatively drier farms"
        })
    (ROOT/"artifacts"/"clustering_report.json").write_text(json.dumps({
        "verification_code": f"AI-A2-{seed:04d}", "silhouette_scores": k_scores,
        "selected_k": selected_k, "profiles": profiles,
        "features_used": FEATURES, "target_columns_excluded": ["yield_tons", "high_risk"]
    }, indent=2))
    # Persist complete pipelines so inference uses identical preprocessing.
    joblib.dump(reg_fitted[selected_reg], ROOT/"models"/"regression_pipeline.joblib")
    joblib.dump(clf_fitted[selected_clf], ROOT/"models"/"classification_pipeline.joblib")
    joblib.dump({"preprocessor": cluster_prep, "model": km, "selected_k": selected_k,
                 "features": FEATURES, "seed": seed}, ROOT/"models"/"clustering_pipeline.joblib")
    # Held-out evidence and plots.
    rpred = reg_fitted[selected_reg].predict(X_test)
    cpred = clf_fitted[selected_clf].predict(X_test)
    cprob = clf_fitted[selected_clf].predict_proba(X_test)[:, list(clf_fitted[selected_clf].classes_).index(1)]
    cluster_test = km.predict(cluster_prep.transform(X_test))
    out = X_test.copy().reset_index(drop=True)
    out.insert(0, "row_id", X_test.index.to_numpy())
    out["actual_yield_tons"] = yr_test.to_numpy()
    out["predicted_yield_tons"] = rpred
    out["yield_residual"] = out["actual_yield_tons"] - out["predicted_yield_tons"]
    out["actual_high_risk"] = yc_test.to_numpy()
    out["predicted_high_risk"] = cpred
    out["high_risk_probability"] = cprob
    out["cluster"] = cluster_test
    out.to_csv(ROOT/"artifacts"/"test_predictions.csv", index=False, float_format="%.4f")
    plots = ROOT/"artifacts"/"plots"
    plt.figure(figsize=(7,4.5))
    plt.scatter(out["predicted_yield_tons"], out["yield_residual"], alpha=.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted yield (tons)"); plt.ylabel("Residual (actual - predicted)")
    plt.title(f"Regression residuals — {selected_reg}"); plt.tight_layout()
    plt.savefig(plots/"residuals.png", dpi=150); plt.close()
    cm = np.array(clf_scores[selected_clf]["confusion_matrix"])
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion matrix — {selected_clf}"); plt.colorbar()
    plt.xticks([0,1], ["Predicted low", "Predicted high"])
    plt.yticks([0,1], ["Actual low", "Actual high"])
    for i in range(2):
        for j in range(2): plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.tight_layout(); plt.savefig(plots/"confusion_matrix.png", dpi=150); plt.close()
    plt.figure(figsize=(7,4.5))
    plt.scatter(Z[:,0], Z[:,1], c=labels, alpha=.8)
    plt.xlabel("Scaled feature component 1"); plt.ylabel("Scaled feature component 2")
    plt.title(f"Farm clusters (visual projection) — K={selected_k}")
    plt.tight_layout(); plt.savefig(plots/"clusters.png", dpi=150); plt.close()
    return {"data_evidence": data_evidence, "regression": reg_scores, "classification": clf_scores,
            "selected_reg": selected_reg, "selected_clf": selected_clf, "selected_k": selected_k,
            "cluster_scores": k_scores, "profiles": profiles, "test_rows": len(X_test)}

def run_all(student_id: str, seed: int):
    if seed != 6559:
        raise ValueError("Seed must equal the last four registration digits: 6559.")
    if not student_id or "@" not in student_id:
        raise ValueError("Provide the student's institutional email with --student-id.")
    from data.generate_data import generate_dataset
    generate_dataset(seed, ROOT/"data"/"AI_A2_2426559.csv")
    result = train_all(seed, student_id)
    (ROOT/"artifacts"/"run_summary.json").write_text(json.dumps({
        "student_id": student_id, "verification_code": f"AI-A2-{seed:04d}",
        "result": result
    }, indent=2))
    print(f"Student ID: {student_id}")
    print(f"Verification code: AI-A2-{seed:04d}")
    print(f"Rows: {result['data_evidence']['input_rows']} input; {result['data_evidence']['clean_rows']} after duplicate removal")
    print(f"Dataset SHA-256: {result['data_evidence']['sha256']}")
    print(f"Selected regression: {result['selected_reg']}; classifier: {result['selected_clf']}; K: {result['selected_k']}")
    print("Artifacts regenerated under artifacts/ and models/.")
    return result
