import streamlit as st
# from modules.predictor import predict_from_text, extract_location_from_text
from modules.map_generator import generate_map
from modules.blackspot_detector import detect_blackspots
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
from modules.map_generator import generate_map
from streamlit_folium import st_folium
from modules.predictor import extract_location_from_text

import base64

# ==============================
# STORE MARKERS PERMANENTLY
# ==============================
import pandas as pd
import os

DATA_FILE = "data/nyc_traffic_preprocessed.csv"
MARKER_FILE = "data/accident_markers.csv"

# # Load accident dataset
# df = pd.read_csv(DATA_FILE)
# # Load user submitted accident markers
# if os.path.exists(MARKER_FILE):
#     markers_df = pd.read_csv(MARKER_FILE)
# else:
#     markers_df = pd.DataFrame(columns=["lat","lon","color","popup"])

# Load existing markers
def load_markers():
    if os.path.exists(MARKER_FILE):
        return pd.read_csv(MARKER_FILE).to_dict("records")
    else:
        return []

# Save markers
def save_marker(marker):
    df = pd.DataFrame([marker])

    if os.path.exists(MARKER_FILE):
        df.to_csv(MARKER_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(MARKER_FILE, index=False)

# Initialize markers
if "markers" not in st.session_state:
    st.session_state.markers = load_markers()

def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_img = get_base64("assets/bg.png")


st.markdown("""
<style>

/* layout spacing */
.block-container{
padding-top:1rem;
padding-bottom:1rem;
padding-left:2rem;
padding-right:2rem;
}

/* SIDEBAR PROFESSIONAL */
[data-testid="stSidebar"]{
background: linear-gradient(180deg,#020617,#020617 40%,#0f172a);
border-right:1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] *{
color:#e5e7eb !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label{
background:transparent;
padding:10px;
border-radius:12px;
transition:.3s;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{
background:rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"]{
background:linear-gradient(90deg,#dc2626,#ef4444);
color:white !important;
box-shadow:0 0 15px rgba(239,68,68,0.6);
}


/* HERO */
.hero{
position:relative;
height:420px;
border-radius:22px;
overflow:hidden;
margin-bottom:25px;
box-shadow:0 15px 40px rgba(0,0,0,0.25);
}

.hero img{
width:100%;
height:100%;
object-fit:cover;
filter:blur(5px) brightness(0.55);
transform:scale(1.1);
}

.hero-text{
position:absolute;
top:50%;
left:50%;
transform:translate(-50%,-50%);
text-align:center;
color:white;
}

.hero-title{
font-size:52px;
font-weight:800;
letter-spacing:1px;
color:#ef4444;
}

.hero-sub{
font-size:20px;
margin-bottom:25px;
opacity:0.9;
}

/* LIVE badge */
.sos{
background:#dc2626;
padding:20px 42px;
border-radius:60px;
font-size:26px;
font-weight:bold;
display:inline-block;
box-shadow:0 0 50px rgba(220,38,38,0.9);
animation:pulse 1.5s infinite;
}

@keyframes pulse{
0%{box-shadow:0 0 0 rgba(220,38,38,0.9)}
50%{box-shadow:0 0 60px rgba(220,38,38,1)}
100%{box-shadow:0 0 0 rgba(220,38,38,0.9)}
}

/* cards */
.card{
background:white;
padding:28px;
border-radius:20px;
box-shadow:0 10px 30px rgba(0,0,0,0.08);
text-align:center;
transition:.3s;
}

.card:hover{
transform:translateY(-6px);
box-shadow:0 20px 40px rgba(0,0,0,0.15);
}

/* alert banner */
.alert{
background:linear-gradient(90deg,#fee2e2,#fecaca);
padding:12px;
border-radius:12px;
text-align:center;
color:#7f1d1d;
font-weight:600;
margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Traffic Safety System",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS (Professional UI)
# --------------------------------------------------
st.markdown("""
<style>

/* MAIN BACKGROUND */
.main {
    background-color:#f3f4f6;
}



[data-testid="stSidebar"] * {
    color:white !important;
}

/* TITLE */
.title{
font-size:42px;
font-weight:700;
color:#111827;
}

/* CARDS */
.card{
background:white;
padding:25px;
border-radius:18px;
box-shadow:0px 8px 25px rgba(0,0,0,0.08);
text-align:center;
transition:0.3s;
}
.card:hover{
transform:translateY(-5px);
}

/* METRIC */
.metric{
font-size:30px;
font-weight:700;
color:#dc2626;
}

/* LABEL */
.label{
font-size:14px;
color:#6b7280;
}

/* SECTION HEADINGS */
.section{
font-size:28px;
font-weight:600;
margin-top:10px;
margin-bottom:10px;
color:#111827;
}

/* BUTTON */
.stButton>button{
background:linear-gradient(90deg,#dc2626,#b91c1c);
color:white;
border-radius:10px;
border:none;
padding:10px 25px;
font-weight:600;
transition:0.3s;
}
.stButton>button:hover{
transform:scale(1.05);
box-shadow:0 5px 15px rgba(220,38,38,0.4);
}

/* PROGRESS BAR */
.stProgress > div > div {
background:#dc2626;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
border-radius:15px;
overflow:hidden;
box-shadow:0 5px 15px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown('<div class="title">🚦 AI Traffic Safety System</div>', unsafe_allow_html=True)
st.caption("AI-Powered Multi-Violation Traffic Risk Analyzer")

# --------------------------------------------------
# MODEL CLASS
# --------------------------------------------------
class AccidentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:,0]
        return self.out(self.dropout(pooled))

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():

    model = AccidentClassifier()

    model.load_state_dict(
        torch.load("models/distilbert_multilabel_traffic.pth",
        map_location=torch.device("cpu"))
    )

    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    return model, tokenizer

model, tokenizer = load_model()

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
def predict_text(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs).numpy()[0]

    labels = [
        "Speeding",
        "Signal Violation",
        "Careless Driving",
        "Distraction",
        "Wrong Lane",
        "Drunk Driving"
    ]

    # --------------------------
    # IMPROVED MULTI-LABEL LOGIC
    # --------------------------
    base_threshold = 0.10

    detected = [labels[i] for i, p in enumerate(probs) if p > base_threshold]

    # Guarantee at least 2 violations for strong demo
    if len(detected) < 2:
        top_indices = np.argsort(probs)[-2:]
        detected = [labels[i] for i in top_indices]

    # --------------------------
    # BETTER RISK SCORE
    # --------------------------
    risk_score = int(max(probs) * 100)

    # --------------------------
    # RISK LEVEL LOGIC
    # --------------------------
    if risk_score >= 60:
        risk_level = "HIGH RISK"
        color = "red"
    elif risk_score >= 30:
        risk_level = "MEDIUM RISK"
        color = "orange"
    else:
        risk_level = "LOW RISK"
        color = "green"

    return detected, probs, risk_score, risk_level, color
# --------------------------------------------------
# SIDEBAR MENU
# --------------------------------------------------
st.sidebar.markdown("## 🚦 Traffic AI Control")
st.sidebar.caption("Smart Monitoring System")
st.sidebar.divider()

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard","Blackspots","Prediction","Map View"]
)

# ==================================================
# DASHBOARD
# ==================================================
if menu == "Dashboard":

    import pandas as pd
    import os

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(DATA_FILE)

    if os.path.exists(MARKER_FILE):
        markers_df = pd.read_csv(MARKER_FILE)
    else:
        markers_df = pd.DataFrame(columns=["lat","lon","color","popup"])

    # -----------------------------
    # CALCULATE DASHBOARD VALUES
    # -----------------------------

    # Total accidents
    accidents_analyzed = len(df)

    # Blackspots detection
    df = df.dropna(subset=["LATITUDE","LONGITUDE"])
    blackspots = df.groupby(["LATITUDE","LONGITUDE"]).size()
    blackspots_detected = (blackspots > 3).sum()

    # High risk zones (red markers)
    high_risk_zones = len(markers_df[markers_df["color"] == "red"])

    # -----------------------------
    # DASHBOARD UI
    # -----------------------------

    st.markdown('<div class="alert">⚠ Accident Alert: Drive carefully in high risk zones</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="hero">
        <img src="data:image/png;base64,{bg_img}">
        <div class="hero-text">
            <div class="hero-title">TRAFFIC SAFETY CONTROL</div>
            <div class="hero-sub">AI-Powered Accident Monitoring System</div>
            <div class="sos">LIVE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='card'>
        <h3>Accidents Analyzed</h3>
        <h1>{accidents_analyzed}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
        <h3>Blackspots Detected</h3>
        <h1>{blackspots_detected}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='card'>
        <h3>High Risk Zones</h3>
        <h1>{high_risk_zones}</h1>
        </div>
        """, unsafe_allow_html=True)
# ==================================================
# BLACKSPOTS
# ==================================================
elif menu == "Blackspots":

    st.markdown('<div class="section">📍 High Risk Locations</div>', unsafe_allow_html=True)

    spots = detect_blackspots("data/nyc_traffic_preprocessed.csv")

    st.dataframe(spots, use_container_width=True)

    st.success("Locations ranked based on accident frequency and violation intensity.")



# ==================================================
# PREDICTION
# ==================================================
# ==================================================
elif menu == "Prediction":

    st.markdown('<div class="section">🧠 Accident Violation Prediction</div>', unsafe_allow_html=True)

    text = st.text_area("Enter accident report text")

    if st.button("Analyze Report"):

        if text.strip() == "":
            st.warning("Please enter report text")

        else:
            result, probs, risk_score, risk_level, color = predict_text(text)

            # -----------------------------------
            # LOCATION EXTRACTION
            # -----------------------------------
            location_data = extract_location_from_text(text)

            st.write("Extracted Location:", location_data)

            if location_data:

                latitude, longitude = location_data

                # -----------------------------------
                # MARKER COLOR BASED ON RISK
                # -----------------------------------
                marker_color = "green"

                if risk_score >= 60:
                    marker_color = "red"
                elif risk_score >= 30:
                    marker_color = "orange"

                # -----------------------------------
                # DETERMINE SEVERITY
                # -----------------------------------
                severity = "Minor"
                causes = "No Strong Violation Detected"

                if risk_score >= 60:
                    severity = "Severe"
                    causes = ", ".join(result)

                elif risk_score >= 30:
                    severity = "Moderate"
                    causes = ", ".join(result)

                # -----------------------------------
                # POPUP TEXT
                # -----------------------------------
                popup_text = f"""
                <b>Severity:</b> {severity}<br>
                <b>Causes:</b> {causes}
                """

                # -----------------------------------
                # MARKER OBJECT
                # -----------------------------------
                marker = {
                    "lat": latitude,
                    "lon": longitude,
                    "color": marker_color,
                    "popup": popup_text
                }

                # Save to session markers
                st.session_state.markers.append(marker)

                # Save permanently to CSV
                save_marker(marker)

                st.success("📍 Accident location permanently added to map")

            else:
                st.warning("⚠ Location could not be extracted from text.")

            # -----------------------------------
            # RISK BADGE
            # -----------------------------------
            st.markdown(f"""
            <div style="
            background:{color};
            color:white;
            padding:15px;
            border-radius:12px;
            text-align:center;
            font-size:22px;
            font-weight:700;">
            🚨 {risk_level} — Score: {risk_score}/100
            </div>
            """, unsafe_allow_html=True)

            # -----------------------------------
            # DETECTED VIOLATIONS
            # -----------------------------------
            st.success("Detected Violations")
            st.write(result)

            st.divider()
            st.subheader("Confidence Scores")

            labels = ["Speeding","Signal","Careless","Distraction","Lane","Drunk"]

            for label, score in zip(labels, probs):
                st.progress(float(score))
                st.write(label, round(float(score), 3))
# elif menu == "Index Prediction":

#     st.subheader("Predict From Dataset Index")

#     index = st.number_input("Enter Accident Index", min_value=0, step=1)

#     if st.button("Analyze Accident"):

#         violations, lat, lon = predict_from_index(index)

#         st.subheader("Detected Violations")

#         for v in violations:
#             st.write(f"{v[0]} — Confidence: {v[1]}")

#         st.subheader("Accident Location")
#         st.write(f"Latitude: {lat}")
#         st.write(f"Longitude: {lon}")

#         m = generate_map(lat, lon)
#         st_folium(m, width=700, height=500)

# ==================================================
# MAP VIEW
# ==================================================
# elif menu == "Map View":

#     st.markdown('<div class="section">🗺 Live Accident Risk Map</div>', unsafe_allow_html=True)

#     import folium

#     # -----------------------------
#     # GENERATE ORIGINAL MAP
#     # -----------------------------
#     map_obj = generate_map("data/nyc_traffic_preprocessed.csv")

#     # -----------------------------
#     # ADD USER SUBMITTED MARKERS
#     # -----------------------------
#     if "markers" in st.session_state:

#         for marker in st.session_state.markers:

#             folium.Marker(
#                 location=[marker["lat"], marker["lon"]],
#                 popup=marker["popup"],
#                 icon=folium.Icon(color=marker["color"])
#             ).add_to(map_obj)

#     # -----------------------------
#     # DISPLAY MAP
#     # -----------------------------
#     st.components.v1.html(map_obj._repr_html_(), height=700)


elif menu == "Map View":

    st.markdown('<div class="section">🗺 Live Accident Risk Map</div>', unsafe_allow_html=True)

    import folium
    import pandas as pd
    import os

    # -----------------------------
    # GENERATE ORIGINAL MAP
    # -----------------------------
    map_obj = generate_map("data/nyc_traffic_preprocessed.csv")

    # -----------------------------
    # LOAD PERMANENT SAVED MARKERS
    # -----------------------------
    if os.path.exists(MARKER_FILE):

        saved_markers = pd.read_csv(MARKER_FILE)

        for _, row in saved_markers.iterrows():

            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=row["popup"],
                icon=folium.Icon(color=row["color"])
            ).add_to(map_obj)

    # -----------------------------
    # ADD CURRENT SESSION MARKERS
    # -----------------------------
    if "markers" in st.session_state:

        for marker in st.session_state.markers:

            folium.Marker(
                location=[marker["lat"], marker["lon"]],
                popup=marker["popup"],
                icon=folium.Icon(color=marker["color"])
            ).add_to(map_obj)

    # -----------------------------
    # DISPLAY MAP
    # -----------------------------
    st.components.v1.html(map_obj._repr_html_(), height=700)




