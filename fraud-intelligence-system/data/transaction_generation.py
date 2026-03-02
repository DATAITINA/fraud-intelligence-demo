import json
import random
import numpy
import uuid
from datetime import datetime, timedelta
from typing import Iterable

_COUNTRIES = ["US", "GB", "NG", "KE", "ZA", "DE", "FR", "IN", "BR", "AE"]
_CITIES = ["Lagos", "London", "Nairobi", "Cape Town", "Berlin", "Paris", "Mumbai", "Sao Paulo", "Dubai", "New York"]
_CURRENCIES = ["USD", "EUR", "GBP", "NGN", "KES", "ZAR", "INR", "BRL", "AED"]
_MERCHANTS = ["Amazon", "Walmart", "Target", "Best Buy", "IKEA", "AliExpress", "Apple", "Netflix", "Uber", "Spotify"]
_CATEGORIES = ["Electronics", "Clothing", "Groceries", "Entertainment", "Travel", "Gambling", "Pharmacy", "Fuel"]
_CHANNELS = ["card_present", "card_not_present", "mobile", "web"]
_CARD_TYPES = ["debit", "credit", "prepaid", "virtual"]
_AUTH_METHODS = ["pin", "otp", "biometric", "none"]
_DEVICE_TYPES = ["ios", "android", "web"]
_ISP = ["comcast", "verizon", "vodacom", "bt", "airtel", "mtn", "vodafone", "orange"]
_MERCHANT_IDS = [
    "merchant_123",
    "merchant_234",
    "merchant_345",
    "merchant_456",
    "merchant_567",
]

def _random_ip() -> str:
    return ".".join(str(random.randint(1, 254)) for _ in range(4))

def _risk_label(features: dict) -> int:
    # Simple synthetic label: higher risk for large amount, risky category,
    # new device, and low account age. This is only for generating training data.
    score = 0.0
    if features["amount"] >= 700:
        score += 0.35
    if features["category"] in {"Electronics", "Entertainment", "Gambling"} and features["amount"] >= 400:
        score += 0.25
    if features["is_new_device"] == 1:
        score += 0.2
    if features["account_age_days"] < 30:
        score += 0.15
    if features["velocity_1h"] >= 5:
        score += 0.15
    if features["previous_chargebacks"] >= 1:
        score += 0.2
    return 1 if score >= 0.6 else 0


def generate_transaction() -> dict:
    now = datetime.now()
    account_age_days = random.randint(1, 2000)
    is_new_device = 1 if random.random() < 0.15 else 0
    velocity_1h = random.randint(0, 10)
    previous_chargebacks = random.randint(0, 3)
    amount = round(random.uniform(5, 1500), 2)
    category = random.choice(_CATEGORIES)
    currency = random.choice(_CURRENCIES)
    merchant = random.choice(_MERCHANTS)
    merchant_id = random.choice(_MERCHANT_IDS)

    tx = {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": (now - timedelta(minutes=random.randint(0, 60 * 24))).isoformat(),
        "amount": amount,
        "currency": currency,
        "merchant": merchant,
        "merchant_id": merchant_id,
        "category": category,
        "channel": random.choice(_CHANNELS),
        "card_type": random.choice(_CARD_TYPES),
        "auth_method": random.choice(_AUTH_METHODS),
        "device_type": random.choice(_DEVICE_TYPES),
        "device_id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        "ip_address": _random_ip(),
        "isp": random.choice(_ISP),
        "country": random.choice(_COUNTRIES),
        "city": random.choice(_CITIES),
        "account_age_days": account_age_days,
        "is_new_device": is_new_device,
        "velocity_1h": velocity_1h,
        "previous_chargebacks": previous_chargebacks,
        "is_international": 1 if currency not in {"USD", "EUR", "GBP"} else 0,
    }
    tx["label"] = _risk_label(tx)
    return tx


def generate_dataset(n: int = 1000) -> list[dict]:
    return [generate_transaction() for _ in range(n)]


def _rng_for_key(key: str) -> random.Random:
    # Deterministic RNG for repeatable investigations.
    return random.Random(key)


def get_transactions_last_hours(merchant_id: str, hours: int = 24, limit: int = 250) -> list[dict]:
    # Simulated transaction database API.
    dataset = generate_dataset(min(limit, 1000))
    cutoff = datetime.now() - timedelta(hours=hours)
    return [
        tx
        for tx in dataset
        if tx["merchant_id"] == merchant_id
        and datetime.fromisoformat(tx["timestamp"]) >= cutoff
    ]


def get_merchant_account(merchant_id: str) -> dict:
    # Simulated merchant account data.
    rng = _rng_for_key(merchant_id)
    return {
        "merchant_id": merchant_id,
        "home_country": rng.choice(_COUNTRIES),
        "risk_tier": rng.choice(["low", "medium", "high"]),
        "kyc_status": rng.choice(["verified", "pending", "restricted"]),
        "payout_hold": rng.choice([True, False, False]),
    }


def get_payment_logs(transaction_id: str) -> dict:
    # Simulated payment logs for a specific transaction.
    rng = _rng_for_key(transaction_id)
    failed_attempts = rng.randint(0, 6)
    return {
        "transaction_id": transaction_id,
        "failed_attempts": failed_attempts,
        "ip_changes": rng.randint(0, 3),
        "device_changes": rng.randint(0, 2),
        "was_3ds_used": rng.choice([True, False]),
    }


def get_banking_status(user_id: str) -> dict:
    # Simulated banking status reports.
    rng = _rng_for_key(user_id)
    return {
        "user_id": user_id,
        "account_status": rng.choice(["active", "limited", "closed"]),
        "recent_returns": rng.randint(0, 2),
        "chargeback_ratio_30d": round(rng.uniform(0, 0.12), 3),
    }


def _explain_suspicion(tx: dict, merchant: dict, logs: dict, banking: dict) -> list[str]:
    reasons = []
    if tx["amount"] >= 1000:
        reasons.append("High transaction amount")
    if tx["is_international"] == 1 or tx["country"] != merchant["home_country"]:
        reasons.append("Unusual country or cross-border payment")
    if tx["velocity_1h"] >= 5:
        reasons.append("High transaction velocity in last hour")
    if tx["previous_chargebacks"] >= 1:
        reasons.append("Prior chargebacks on the account")
    if tx["category"] in {"Gambling", "Electronics"} and tx["amount"] >= 400:
        reasons.append("Risky category with elevated amount")
    if tx["is_new_device"] == 1:
        reasons.append("New device for this user")
    if tx["auth_method"] == "none":
        reasons.append("No authentication method used")
    if logs["failed_attempts"] >= 3:
        reasons.append("Multiple failed payment attempts before success")
    if logs["ip_changes"] >= 2:
        reasons.append("Multiple IP changes during payment flow")
    if banking["account_status"] in {"limited", "closed"}:
        reasons.append("Banking account status is not active")
    if banking["chargeback_ratio_30d"] >= 0.05:
        reasons.append("Elevated chargeback ratio in last 30 days")
    if merchant["risk_tier"] == "high" or merchant["kyc_status"] != "verified":
        reasons.append("Merchant risk tier or KYC status is concerning")
    return reasons


def _suggest_actions(reasons: Iterable[str]) -> list[str]:
    actions = []
    reason_set = set(reasons)
    if "High transaction amount" in reason_set:
        actions.append("Verify customer identity and funding source")
    if "Unusual country or cross-border payment" in reason_set:
        actions.append("Confirm customer location and device/IP consistency")
    if "Multiple failed payment attempts before success" in reason_set:
        actions.append("Review authentication logs and attempt patterns")
    if "New device for this user" in reason_set:
        actions.append("Trigger step-up authentication for future payments")
    if "Banking account status is not active" in reason_set:
        actions.append("Hold payout and contact banking partner for details")
    if "Merchant risk tier or KYC status is concerning" in reason_set:
        actions.append("Escalate to merchant onboarding/AML review")
    if not actions:
        actions.append("Manual review by fraud analyst")
    return actions


def find_suspicious_transactions(transactions: list[dict]) -> list[dict]:
    suspicious = []
    for tx in transactions:
        merchant = get_merchant_account(tx["merchant_id"])
        logs = get_payment_logs(tx["transaction_id"])
        banking = get_banking_status(tx["user_id"])
        reasons = _explain_suspicion(tx, merchant, logs, banking)
        if reasons:
            suspicious.append(
                {
                    "transaction_id": tx["transaction_id"],
                    "amount": tx["amount"],
                    "currency": tx["currency"],
                    "merchant_id": tx["merchant_id"],
                    "reason": "; ".join(reasons),
                    "suggested_action": "; ".join(_suggest_actions(reasons)),
                }
            )
    return suspicious


def investigation_summary(transactions: list[dict]) -> list[dict]:
    # Structured summaries for reporting.
    return find_suspicious_transactions(transactions)


def render_json_report(summaries: list[dict]) -> str:
    return json.dumps(summaries, indent=2, sort_keys=True)


def render_markdown_table(summaries: list[dict]) -> str:
    if not summaries:
        return "| transaction_id | amount | currency | reason | suggested_action |\n|---|---:|---|---|---|\n"
    header = "| transaction_id | amount | currency | reason | suggested_action |\n|---|---:|---|---|---|\n"
    rows = [
        f"| {s['transaction_id']} | {s['amount']:.2f} | {s['currency']} | {s['reason']} | {s['suggested_action']} |"
        for s in summaries
    ]
    return header + "\n".join(rows)


def run_fraud_investigation(merchant_id: str, hours: int = 24, limit: int = 250) -> dict:
    # Main entry point for the Fraud Investigation Copilot.
    transactions = get_transactions_last_hours(merchant_id, hours=hours, limit=limit)
    summaries = investigation_summary(transactions)
    return {
        "merchant_id": merchant_id,
        "hours": hours,
        "count_reviewed": len(transactions),
        "count_flagged": len(summaries),
        "summaries": summaries,
    }


if __name__ == "__main__":
    for _ in range(10):
        print(generate_transaction())
