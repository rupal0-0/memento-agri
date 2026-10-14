"""
Case bank construction for Memento-Agri.

Converts tabular agricultural-climate observations and outcomes into structured
historical memory cases for outcome-aware retrieval.
"""
from pathlib import Path
import json
import os
import pandas as pd
import numpy as np
from config import RANDOM_STATE


def safe_val(val, default=None):
    """Clean pandas/numpy values for clean JSON serialization."""
    if val is None or pd.isna(val):
        return default
    if hasattr(val, "item"):
        return val.item()
    return val


def build_case_bank(df: pd.DataFrame, output_path: str = None, max_cases: int = None) -> list:
    """
    Build a structured case bank from the preprocessed dataframe.
    Selects a balanced, representative set of cases across states and crops.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed agricultural dataset.
    output_path : str, optional
        Path where the cases JSON file should be saved.
    max_cases : int, optional
        Maximum number of cases to index in memory. If None, reads from
        environment variable MAX_CASES (defaults to 5000 for responsive memory retrieval).
        Set to 0 or 'all' to include every record.

    Returns
    -------
    list
        List of structured case dictionaries.
    """
    if max_cases is None:
        env_val = os.getenv("MAX_CASES", "5000")
        max_cases = 0 if env_val.lower() == "all" else int(env_val)

    if max_cases > 0 and len(df) > max_cases:
        # Stratified sampling across (state, crop) pairs so every state and crop is well-represented
        groups = df.groupby(["state", "crop"])
        n_groups = len(groups)
        samples_per_group = max(1, max_cases // max(1, n_groups))
        sampled_df = (
            df.groupby(["state", "crop"], group_keys=False)
            .apply(lambda g: g.sample(min(len(g), samples_per_group), random_state=RANDOM_STATE))
        )
        if len(sampled_df) < max_cases:
            remaining = df.drop(sampled_df.index)
            needed = max_cases - len(sampled_df)
            sampled_df = pd.concat([sampled_df, remaining.sample(min(len(remaining), needed), random_state=RANDOM_STATE)])
        sampled_df = sampled_df.head(max_cases).sort_values(by=["year", "state", "crop"]).reset_index(drop=True)
    else:
        sampled_df = df.copy()

    cases = []
    
    for idx, row in sampled_df.iterrows():
        case_id = str(row.get("case_id", f"C{idx + 1:07d}"))
        state = str(safe_val(row.get("state"), "Unknown"))
        district = str(safe_val(row.get("district"), "Unknown"))
        crop = str(safe_val(row.get("crop"), "Unknown"))
        year = int(safe_val(row.get("year"), 0))
        season = str(safe_val(row.get("season"), "Kharif/Rabi"))

        # Climate variables
        temp = float(safe_val(row.get("temperature"), 0.0))
        humidity = float(safe_val(row.get("humidity"), 0.0))
        rainfall = float(safe_val(row.get("rainfall"), 0.0))
        wind_speed = float(safe_val(row.get("wind_speed"), 0.0))
        solar = float(safe_val(row.get("solar_radiation"), 0.0))
        ph = float(safe_val(row.get("ph"), 7.0))
        real_rain = float(safe_val(row.get("rainfall_real_mm"), rainfall))

        # Anomalies & Stress
        rain_anom = float(safe_val(row.get("rainfall_anomaly"), 0.0))
        temp_anom = float(safe_val(row.get("temperature_anomaly"), 0.0))
        hum_anom = float(safe_val(row.get("humidity_anomaly"), 0.0))
        wind_anom = float(safe_val(row.get("wind_anomaly"), 0.0))
        solar_anom = float(safe_val(row.get("solar_anomaly"), 0.0))

        # Risk
        risk_score = float(safe_val(row.get("climate_risk"), 0.0))
        risk_cat = str(safe_val(row.get("risk_category"), "Unknown"))

        # Outcome
        crop_yield = float(safe_val(row.get("crop_yield"), 0.0))
        yield_baseline = float(safe_val(row.get("yield_baseline"), crop_yield))
        yield_anom = float(safe_val(row.get("yield_anomaly"), 0.0))

        # Soil & Irrigation
        soil_type = str(safe_val(row.get("soil_type"), "Loam / Regional Soil"))
        irrigation_pct = float(safe_val(row.get("irrigation_percentage"), 0.0))
        water_stress = float(safe_val(row.get("rainfall_stress"), 0.0))
        moisture_def = float(safe_val(row.get("humidity_stress"), 0.0))

        # Human-readable summary for semantic embedding
        summary_text = (
            f"State: {state}, District: {district}, Crop: {crop}, Year: {year}, Season: {season}. "
            f"Temperature: {temp:.1f}°C (anomaly: {temp_anom:+.1%}), "
            f"Rainfall: {real_rain:.1f} mm (anomaly: {rain_anom:+.1%}), "
            f"Humidity: {humidity:.1f}%. "
            f"Climate Risk: {risk_cat} ({risk_score:.2f}). "
            f"Yield: {crop_yield:.1f} kg/ha (anomaly: {yield_anom:+.1%})."
        )

        case = {
            "case_id": case_id,
            "state": state,
            "district": district,
            "crop": crop,
            "year": year,
            "season": season,
            "temperature": temp,
            "humidity": humidity,
            "rainfall": real_rain,
            "wind_speed": wind_speed,
            "solar_radiation": solar,
            "ph": ph,
            "rainfall_anomaly": rain_anom,
            "temperature_anomaly": temp_anom,
            "humidity_anomaly": hum_anom,
            "wind_anomaly": wind_anom,
            "solar_anomaly": solar_anom,
            "climate_risk": risk_score,
            "risk_category": risk_cat,
            "crop_yield": crop_yield,
            "yield_baseline": yield_baseline,
            "yield_anomaly": yield_anom,
            "context": {
                "state": state,
                "district": district,
                "crop": crop,
                "year": year,
                "season": season,
                "soil_type": soil_type,
                "rainfall_anomaly": rain_anom,
                "temperature_anomaly": temp_anom,
                "water_stress": water_stress,
                "moisture_deficit": moisture_def,
                "irrigation_percentage": irrigation_pct,
                "temperature": temp,
                "humidity": humidity,
                "rainfall": real_rain,
                "climate_risk": risk_score,
            },
            "risk": {
                "score": risk_score,
                "category": risk_cat,
            },
            "outcome": {
                "crop_yield": crop_yield,
                "yield_baseline": yield_baseline,
                "yield_anomaly": yield_anom,
                "positive_outcome": bool(yield_anom >= 0),
            },
            "text": summary_text,
        }
        cases.append(case)

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2)

    return cases
