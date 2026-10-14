"""
Climate risk classification model for Memento-Agri.

Predicts climate risk category (Low, Moderate, High, Severe)
based on observed climate variables and anomalies.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from config import MODEL_DIR, RANDOM_STATE


def train_risk_model(df: pd.DataFrame, model_path: str = None) -> dict:
    """
    Train a climate risk classification model.

    Parameters
    ----------
    df : pd.DataFrame
        Processed agricultural dataset.
    model_path : str, optional
        File path to save the trained model.

    Returns
    -------
    dict
        Evaluation metrics and training summary.
    """
    if "risk_category" not in df.columns:
        raise ValueError("Target column 'risk_category' not found in dataset.")

    candidate_features = [
        "temperature", "humidity", "ph", "rainfall", "wind_speed", "solar_radiation",
        "rainfall_anomaly", "temperature_anomaly", "humidity_anomaly", "wind_anomaly", "solar_anomaly",
        "rainfall_stress", "temperature_stress", "humidity_stress", "wind_stress", "solar_stress",
        "climate_risk"
    ]
    features = [c for c in candidate_features if c in df.columns]

    # Clean data
    clean_df = df.dropna(subset=features + ["risk_category"]).copy()
    X = clean_df[features]
    y = clean_df["risk_category"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    out_path = Path(model_path) if model_path else MODEL_DIR / "risk_model.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    pipeline = {
        "model": clf,
        "features": features,
        "classes": list(clf.classes_),
    }
    joblib.dump(pipeline, str(out_path))

    return {
        "status": "success",
        "model_type": "RandomForestClassifier",
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
        "features_used": features,
        "classes": list(clf.classes_),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "saved_model": str(out_path)
    }
