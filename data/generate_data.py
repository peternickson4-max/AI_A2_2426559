
"""Generate the personalized, deterministic synthetic potato harvest dataset."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg",
    "field_size_ha", "season", "cooperative_support",
]
ALL_COLUMNS = FEATURES + ["yield_tons", "high_risk"]

def verification_code(seed: int) -> str:
    return f"AI-A2-{seed:04d}"

def generate_dataset(seed: int, output_path: Path | None = None) -> pd.DataFrame:
    if not 0 <= seed <= 9999:
        raise ValueError("Seed must be the last four registration digits (0000–9999).")
    rng = np.random.default_rng(seed)
    n = 240 + (seed % 41)
    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    moisture = rng.uniform(20, 80, n)
    fertilizer = rng.uniform(0, 250, n)
    size = rng.uniform(0.2, 4.0, n)
    season = rng.choice(np.array(["A", "B"]), size=n)
    support = rng.integers(0, 2, size=n)
    season_effect = np.where(season == "A", 0.50, -0.20)
    noise = rng.normal(0, 0.80, n)
    y = (0.004 * rainfall + 0.06 * moisture + 0.02 * fertilizer
         + 1.30 * size - 0.25 * np.abs(temp - 18)
         + 0.70 * support + season_effect + noise)
    y = np.clip(y, 0.50, None)
    df = pd.DataFrame({
        "rainfall_mm": rainfall, "avg_temp_c": temp,
        "soil_moisture_pct": moisture, "fertilizer_kg": fertilizer,
        "field_size_ha": size, "season": season,
        "cooperative_support": support, "yield_tons": y,
    })
    yield_per_ha = df["yield_tons"] / df["field_size_ha"]
    df["high_risk"] = (yield_per_ha < yield_per_ha.median()).astype(int)
    m = round(0.05 * n)
    for col in ("soil_moisture_pct", "fertilizer_kg"):
        positions = rng.choice(n, size=m, replace=False)
        df.loc[positions, col] = np.nan
    df = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, float_format="%.4f")
        fingerprint = hashlib.sha256(output_path.read_bytes()).hexdigest()
        fp_path = ROOT / "artifacts" / "data_fingerprint.txt"
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        fp_path.write_text(
            f"Verification code: {verification_code(seed)}\n"
            f"Seed: {seed}\nCSV: {output_path.relative_to(ROOT)}\n"
            f"SHA-256: {fingerprint}\n", encoding="utf-8"
        )
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "AI_A2_2426559.csv")
    args = parser.parse_args()
    df = generate_dataset(args.seed, args.output)
    print(f"Verification code: {verification_code(args.seed)}")
    print(f"Rows including deliberate duplicates: {len(df)}")
    print(f"CSV written to: {args.output}")
    print(f"SHA-256: {hashlib.sha256(args.output.read_bytes()).hexdigest()}")

if __name__ == "__main__":
    main()
