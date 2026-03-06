
# Global variable to store model
_model = None


def load_model():
    global _model

    if _model is None:
        try:
            _model = torch.load("model/trained_model.pt", map_location="cpu")
            _model.eval()
        except:
            _model = None

    return _model


def predict(text, model):
    """
    Dummy prediction logic (until Member-2 gives model)
    """

    # Simple demo logic
    text_lower = text.lower()

    violations = []
    severity = "Low"
    score = 30
    location = "Unknown"

    if "speed" in text_lower:
        violations.append("Overspeeding")
        severity = "High"
        score = 85

    if "drunk" in text_lower:
        violations.append("Drunk Driving")
        severity = "High"
        score = 90

    if "signal" in text_lower:
        violations.append("Signal Violation")

    if "nh" in text_lower:
        location = "NH-45"

    if not violations:
        violations = ["Careless Driving"]

    return {
        "violations": violations,
        "severity": severity,
        "score": score,
        "location": location
    }


import pandas as pd
from datetime import datetime
import os


def save_result(result):

    file_path = "data/predictions.csv"

    row = {
        "date": datetime.now(),
        "location": result["location"],
        "severity": result["severity"],
        "violation": ",".join(result["violations"])
    }

    df = pd.DataFrame([row])

    # Always write header if file doesn't exist
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode="a", header=False, index=False)


def detect_blackspots(threshold=3):
    """
    Detect locations with accidents above threshold
    """

    file_path = "data/predictions.csv"

    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)

    if df.empty:
        return None

    # Count accidents per location
    location_counts = df["location"].value_counts().reset_index()
    location_counts.columns = ["location", "accident_count"]

    # Filter black spots
    blackspots = location_counts[
        location_counts["accident_count"] >= threshold
    ]

    return blackspots

