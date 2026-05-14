from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import numpy as np
import joblib
import os
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# =========================
# Load Models
# =========================

kmeans = joblib.load(os.path.join(ARTIFACTS_DIR, "kmeans_model.pkl"))

iso_vip = joblib.load(os.path.join(ARTIFACTS_DIR, "iso_vip.pkl"))
iso_regular = joblib.load(os.path.join(ARTIFACTS_DIR, "iso_reg.pkl"))
iso_occ = joblib.load(os.path.join(ARTIFACTS_DIR, "iso_occ.pkl"))


# =========================
# Labels & Rules
# =========================

CLUSTER_LABELS = {
    0: "At-Risk / Lost Customers",
    1: "Regular Customers",
    2: "Occasional Buyers",
    3: "VIP / High-Value Customers"
}

DECISION_RULES = {
    3: "Manual Review (High Risk)",
    2: "Upsell / Growth Opportunity",
    1: "Monitor Behavior",
    0: "No Action (Ignore)"
}


# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="Customer Segmentation & Anomaly Detection API",
    description="KMeans segmentation + Isolation Forest anomaly detection",
    version="1.0.0"
)


# =========================
# Request Schema + Strong Validation
# =========================

class CustomerInput(BaseModel):
    recency: float = Field(
        ...,
        ge=0,
        description="Days since last purchase. Must be >= 0",
        example=30
    )

    frequency: float = Field(
        ...,
        ge=0,
        description="Number of purchases. Must be >= 0",
        example=3
    )

    monetary: float = Field(
        ...,
        ge=0,
        description="Total customer spend. Must be >= 0",
        example=1200
    )

    @field_validator("recency", "frequency", "monetary")
    @classmethod
    def value_must_be_finite(cls, value):
        if value is None:
            raise ValueError("Value cannot be None")

        if not math.isfinite(value):
            raise ValueError("Value must be finite. NaN or Infinity is not allowed")

        return value


# =========================
# Helper Functions
# =========================

def validate_business_rules(recency: float, frequency: float, monetary: float):
    """
    Business-level validation beyond type/range validation.
    """

    # Case: no purchase customer
    if frequency == 0 and monetary == 0:
        return {
            "valid": True,
            "type": "new_customer",
            "message": "No purchase / new customer"
        }

    # Inconsistent cases
    if frequency == 0 and monetary > 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid input: frequency is 0 but monetary is positive"
        )

    if frequency > 0 and monetary == 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid input: monetary is 0 but frequency is positive"
        )

    # Optional sanity limits
    if recency > 5000:
        raise HTTPException(
            status_code=400,
            detail="Invalid input: recency is unrealistically high"
        )

    if frequency > 10000:
        raise HTTPException(
            status_code=400,
            detail="Invalid input: frequency is unrealistically high"
        )

    if monetary > 100_000_000:
        raise HTTPException(
            status_code=400,
            detail="Invalid input: monetary is unrealistically high"
        )

    return {
        "valid": True,
        "type": "normal_customer",
        "message": "Valid input"
    }


def preprocess_customer(recency: float, frequency: float, monetary: float):
    """
    Preprocess customer input using the same logic used during training.
    AvgOrderValue is computed automatically.
    """

    avg_order_value = monetary / frequency

    X = np.array([[
        recency,
        frequency,
        np.log1p(monetary),
        np.log1p(avg_order_value)
    ]])

    return X, avg_order_value


def detect_anomaly(segment_id: int, X: np.ndarray) -> bool:
    """
    Segment-aware anomaly detection.
    """

    if segment_id == 3:
        return bool(iso_vip.predict(X)[0] == -1)

    if segment_id == 1:
        return bool(iso_regular.predict(X)[0] == -1)

    if segment_id == 2:
        return bool(iso_occ.predict(X)[0] == -1)

    # No anomaly detection for At-Risk segment
    return False


# =========================
# Endpoints
# =========================

@app.get("/")
def home():
    return {
        "message": "Customer Segmentation & Anomaly Detection API is running",
        "docs": "/docs"
    }


@app.post("/predict")
def predict_customer(data: CustomerInput):

    recency = float(data.recency)
    frequency = float(data.frequency)
    monetary = float(data.monetary)

    # Business validation
    validation_result = validate_business_rules(
        recency=recency,
        frequency=frequency,
        monetary=monetary
    )

    # New / no purchase customer
    if validation_result["type"] == "new_customer":
        return {
            "status": "ok",
            "segment_id": None,
            "segment": "No Purchase / New Customer",
            "avg_order_value": 0.0,
            "anomaly": False,
            "recommended_action": "Ignore",
            "message": validation_result["message"]
        }

    # Preprocess
    X, avg_order_value = preprocess_customer(
        recency=recency,
        frequency=frequency,
        monetary=monetary
    )

    # Segmentation
    segment_id = int(kmeans.predict(X)[0])
    segment_label = CLUSTER_LABELS.get(segment_id, "Unknown Segment")

    # Anomaly Detection
    anomaly = detect_anomaly(segment_id, X)

    # Decision
    action = DECISION_RULES.get(segment_id, "No Action") if anomaly else "No Action"

    return {
        "status": "ok",
        "segment_id": segment_id,
        "segment": segment_label,
        "avg_order_value": round(float(avg_order_value), 2),
        "anomaly": bool(anomaly),
        "recommended_action": action,
        "message": validation_result["message"]
    }