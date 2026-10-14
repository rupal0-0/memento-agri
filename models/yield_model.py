"""
Crop yield prediction model for Memento-Agri.

Trains a predictive regressor to forecast crop yield under observed climate
and nutrient conditions, providing baseline comparisons for the outcome memory layer.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from config import MODEL_DIR, RANDOM_STATE


def train_yield_model(df: pd.DataFrame, model_path: str = None) -> dict:
    """
    Train a crop yield regression model.

    Parameters
    ----------
    df : pd.DataFrame
        Processed agricultural dataset.
    model_path : str, optional
        File path to save the trained model.

    Returns
    -------
    dict
        Evaluation metrics (MAE, RMSE, R2) and training summary.
    """
    if "crop_yield" not in df.columns:
        raise ValueError("Target column 'crop_yield' not found in dataset.")

    cat_cols = [c for c in ["crop", "state"] if c in df.columns]
    num_cols = [
        c for c in [
            "crop_area", "n_req", "p_req", "k_req", "total_n", "total_p", "total_k",
            "temperature", "humidity", "ph", "rainfall", "wind_speed", "solar_radiation",
            "rainfall_anomaly", "temperature_anomaly", "humidity_anomaly", "wind_anomaly", "solar_anomaly",
            "climate_risk"
        ] if c in df.columns
    ]

    all_features = cat_cols + num_cols
    clean_df = df.dropna(subset=all_features + ["crop_yield"]).copy()

    X = clean_df[all_features]
    y = clean_df["crop_yield"].astype(float)

    # Use temporal or stratified split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    regressor = HistGradientBoostingRegressor(
        max_iter=150,
        random_state=RANDOM_STATE
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    out_path = Path(model_path) if model_path else MODEL_DIR / "yield_model.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, str(out_path))

    return {
        "status": "success",
        "model_type": "HistGradientBoostingRegressor",
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
        "categorical_features": cat_cols,
        "numerical_features": num_cols,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "saved_model": str(out_path)
    }
