from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from datetime import datetime

def _timestamp_features(ts: str) -> tuple[int, int]:
    dt = datetime.fromisoformat(ts)
    return dt.hour, dt.weekday()

def _prepare_features(tx:dict, columns: list[str]) -> np.ndarray:
    df = pd.DataFrame([tx])
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    if "timestamp" in df.columns and pd.notna(df['timestamp'].iloc[0]):
        hours, weekdays = zip(*df["timestamp"].map(_timestamp_features))
        df["tx_hour"] = list(hours)
        df["tx_weekday"] = list(weekdays)
        df = df.drop(columns=["timestamp"])

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.transaction_generation import generate_transaction  # noqa: E402
from model.train import FraudNet  # noqa: E402


def _load_columns() -> list[str]:
    with open(Path(__file__).parent / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)["columns"]


def _prepare_features(tx: dict, columns: list[str]) -> np.ndarray:
    df = pd.DataFrame([tx])
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    X_num = df[numeric_cols]
    X_cat = pd.get_dummies(df[cat_cols], drop_first=False)
    X_all = pd.concat([X_num, X_cat], axis=1)

    # Align to training columns.
    for col in columns:
        if col not in X_all.columns:
            X_all[col] = 0
    X_all = X_all[columns]
    return X_all.to_numpy(dtype=np.float32)


def predict(tx: dict) -> float:
    columns = _load_columns()
    x = _prepare_features(tx, columns)
    model = FraudNet(input_dim=len(columns))
    state = torch.load(Path(__file__).parent / "fraud_net.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        prob = torch.sigmoid(logits).item()
    return prob


if __name__ == "__main__":
    tx = generate_transaction()
    score = predict(tx)
    print({"transaction_id": tx["transaction_id"], "risk_score": score})
