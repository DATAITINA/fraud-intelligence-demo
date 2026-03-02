from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Allow running this file directly without PYTHONPATH gymnastics.
    sys.path.insert(0, str(ROOT))

from data.transaction_generation import generate_transaction  # noqa: E402
from model.infer import predict  # noqa: E402

app = FastAPI(title="Fraud Intelligence API", version="0.2.0")


class TransactionIn(BaseModel):
    amount: float = Field(..., ge=0)
    currency: str
    merchant: str | None = None
    category: str
    timestamp: str | None = None


class ScoreOut(BaseModel):
    risk_score: float
    is_fraud: bool
    reasons: list[str]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/transactions/generate")
def generate() -> dict[str, Any]:
    return generate_transaction()


@app.post("/transactions/score", response_model=ScoreOut)
def score(tx: TransactionIn) -> ScoreOut:
    # Start from a full synthetic transaction so all expected fields exist.
    payload = generate_transaction()
    payload.update(tx.model_dump())

    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()

    try:
        risk_score = float(predict(payload))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run `python model/train.py` first.",
        ) from exc

    reasons: list[str] = []
    if risk_score >= 0.7:
        reasons.append("model_score_high")

    return ScoreOut(
        risk_score=risk_score,
        is_fraud=risk_score >= 0.7,
        reasons=reasons,
    )
