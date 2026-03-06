
import torch
import pandas as pd
from transformers import DistilBertTokenizer
from model_loader import load_model
# from geopy.geocoders import Nominatim

# =====================================================
# DEVICE SETUP
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# TOKENIZER (Safe to load at import)
# =====================================================
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# =====================================================
# LAZY MODEL LOADING (IMPORTANT FIX)
# =====================================================
model = None

def get_model():
    global model
    if model is None:
        model = load_model()
        model.to(device)
        model.eval()
    return model

# =====================================================
# LABELS
# =====================================================
labels = [
    "speeding",
    "signal_violation",
    "careless_driving",
    "distracted",
    "wrong_lane",
    "drink_drive"
]

# =====================================================
# LOAD DATASET (Used for index-based prediction)
# =====================================================
df = pd.read_csv("data/nyc_traffic_preprocessed.csv")

# =====================================================
# 🔥 LOCATION EXTRACTION FUNCTION
# =====================================================

import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


def extract_location_from_text(text):

    geolocator = Nominatim(user_agent="traffic_ai_system")

    # ROAD PATTERN
    road_pattern = r'\b([A-Z][a-zA-Z0-9\-]*(?:\s[A-Z][a-zA-Z0-9\-]*)*\s(?:Road|Street|Avenue|Boulevard|Parkway|Drive|Lane|Expressway|Highway))\b'
    road_match = re.search(road_pattern, text)

    # BOROUGH
    borough_pattern = r'\b(Queens|Brooklyn|Bronx|Manhattan|Staten Island)\b'
    borough_match = re.search(borough_pattern, text)

    # NEIGHBORHOOD
    neighborhood_pattern = r'\b(Harlem|Flushing|Astoria|Williamsburg|Bushwick|Jamaica|Fordham|Chelsea|SoHo|Tribeca|Midtown|Upper East Side|Upper West Side|Bedford-Stuyvesant|Financial District|Chinatown|Greenwich Village|Long Island City)\b'
    neighborhood_match = re.search(neighborhood_pattern, text)

    location_query = None

    if road_match and borough_match:
        location_query = road_match.group(1) + ", " + borough_match.group(1) + ", New York"

    elif road_match:
        location_query = road_match.group(1) + ", New York"

    elif neighborhood_match:
        location_query = neighborhood_match.group(1) + ", New York"

    elif borough_match:
        location_query = borough_match.group(1) + ", New York"

    if location_query:

        location = geolocator.geocode(location_query)

        if location:
            return (location.latitude, location.longitude)

    return None
# =====================================================
# INDEX-BASED PREDICTION (Dataset Based)
# =====================================================
def predict_from_index(index):

    model = get_model()

    row = df.iloc[index]
    text = row["text_features"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0]

    threshold = 0.25
    predicted = []

    for i, prob in enumerate(probs):
        if prob.item() > threshold:
            predicted.append((labels[i], round(prob.item(), 3)))

    # Fallback if nothing crosses threshold
    if len(predicted) == 0:
        top_indices = torch.topk(probs, 2).indices
        for idx in top_indices:
            predicted.append((labels[idx], round(probs[idx].item(), 3)))

    latitude = row["LATITUDE"]
    longitude = row["LONGITUDE"]

    return predicted, latitude, longitude

# =====================================================
# 🔥 USER TEXT-BASED PREDICTION
# =====================================================

def predict_from_text(text):
    model = get_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs)[0]

    threshold = 0.25
    predicted = []

    for i, prob in enumerate(probs):
        if prob.item() > threshold:
            predicted.append((labels[i], round(prob.item(), 3)))

    if len(predicted) == 0:
        top_indices = torch.topk(probs, 2).indices
        for idx in top_indices:
            predicted.append((labels[idx], round(probs[idx].item(), 3)))

    # 🔥 NEW PART — Find similar dataset row
    text_lower = text.lower()

    for _, row in df.iterrows():
        if any(word in row["text_features"] for word in text_lower.split()):
            return predicted, row["LATITUDE"], row["LONGITUDE"]

    # fallback — use NYC center
    return predicted, 40.7128, -74.0060