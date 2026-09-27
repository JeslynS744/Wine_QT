"""
app.py — Wine Quality Predictor — Streamlit Web App
=====================================================
Loads the trained RandomForest model and StandardScaler from disk,
accepts user inputs for 11 physicochemical features, and predicts
the quality score (3–8) with a qualitative label.
"""

import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍷 Wine Quality Predictor",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a polished look ────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background gradient */
.stApp {
    background: linear-gradient(135deg, #1a0a0a 0%, #2d0f1a 40%, #1a1a2e 100%);
    min-height: 100vh;
}

/* Hero header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(139,0,0,0.25) 0%, rgba(180,30,60,0.15) 100%);
    border-radius: 20px;
    border: 1px solid rgba(220,60,80,0.3);
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}
.hero-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #ff6b8a, #ff9a5c, #ffd460);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-header p {
    color: rgba(255,255,255,0.65);
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e0a12 0%, #12121e 100%);
    border-right: 1px solid rgba(220,60,80,0.2);
}
[data-testid="stSidebar"] .stSlider > div {
    color: #f0e0e8;
}

/* Result cards */
.result-card {
    padding: 1.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-top: 1.5rem;
    font-family: 'Inter', sans-serif;
}
.result-bad {
    background: linear-gradient(135deg, rgba(180,30,30,0.35), rgba(120,20,20,0.2));
    border: 2px solid rgba(220,60,60,0.5);
}
.result-average {
    background: linear-gradient(135deg, rgba(180,120,0,0.35), rgba(140,90,0,0.2));
    border: 2px solid rgba(220,180,0,0.5);
}
.result-good {
    background: linear-gradient(135deg, rgba(20,160,80,0.35), rgba(10,120,50,0.2));
    border: 2px solid rgba(40,200,100,0.5);
}
.result-score {
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-label {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.result-desc {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.7);
}

/* Predict button */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #8b0000, #c0392b, #e74c3c);
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 0.75rem 2rem;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.3s ease;
    letter-spacing: 0.03em;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(220,60,60,0.45);
}

/* Section headers inside sidebar */
.sidebar-section {
    color: #ff9ab0;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1.2rem 0 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ── Auto-train if .pkl files are missing (e.g. first deploy on Render) ────────
def auto_train():
    """Train and save model + scaler if they don't exist on disk."""
    model_path  = os.path.join(os.path.dirname(__file__), "wine_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return  # already trained — nothing to do

    csv_path = os.path.join(os.path.dirname(__file__), "WineQT.csv")
    if not os.path.exists(csv_path):
        return  # dataset not available — let load_artifacts() surface the error

    # Load & prepare data
    df = pd.read_csv(csv_path)
    df.drop(columns=["Id"], inplace=True, errors="ignore")
    X = df.drop(columns=["quality"])
    y = df["quality"]

    # Split
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train
    model = RandomForestClassifier(
        n_estimators=200, min_samples_split=4, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Save
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)


# Run auto-train before anything else (no-op if .pkl files already exist)
auto_train()


# ── Load model and scaler ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load and cache the trained model and scaler from disk."""
    model_path  = os.path.join(os.path.dirname(__file__), "wine_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🍷 Wine Quality Predictor</h1>
    <p>Adjust the physicochemical properties in the sidebar and click Predict to evaluate wine quality.</p>
</div>
""", unsafe_allow_html=True)

# ── Guard: check model loaded ─────────────────────────────────────────────────
if model is None or scaler is None:
    st.error(
        "⚠️ Model files not found!\n\n"
        "Please run `python train.py` first to generate `wine_model.pkl` and `scaler.pkl`."
    )
    st.stop()

# ── Sidebar — Feature Inputs ──────────────────────────────────────────────────
st.sidebar.markdown("## 🧪 Physicochemical Inputs")
st.sidebar.markdown("Drag the sliders to set each feature value.")

st.sidebar.markdown('<p class="sidebar-section">Acidity</p>', unsafe_allow_html=True)
fixed_acidity    = st.sidebar.slider("Fixed Acidity",       min_value=4.6,  max_value=15.9, value=8.3,   step=0.1,  help="g/dm³ — tartaric acid equivalents")
volatile_acidity = st.sidebar.slider("Volatile Acidity",    min_value=0.12, max_value=1.58, value=0.53,  step=0.01, help="g/dm³ — acetic acid (too high → vinegar taste)")
citric_acid      = st.sidebar.slider("Citric Acid",         min_value=0.0,  max_value=1.0,  value=0.27,  step=0.01, help="g/dm³ — adds freshness")

st.sidebar.markdown('<p class="sidebar-section">Sugar & Salt</p>', unsafe_allow_html=True)
residual_sugar   = st.sidebar.slider("Residual Sugar",      min_value=0.9,  max_value=15.5, value=2.5,   step=0.1,  help="g/dm³ — sugar remaining after fermentation")
chlorides        = st.sidebar.slider("Chlorides",           min_value=0.012,max_value=0.611,value=0.087, step=0.001,help="g/dm³ — salt in wine")

st.sidebar.markdown('<p class="sidebar-section">Sulfur Dioxide</p>', unsafe_allow_html=True)
free_sulfur_dio  = st.sidebar.slider("Free Sulfur Dioxide", min_value=1.0,  max_value=68.0, value=15.6,  step=0.5,  help="mg/dm³ — prevents microbial growth")
total_sulfur_dio = st.sidebar.slider("Total Sulfur Dioxide",min_value=6.0,  max_value=289.0,value=45.9,  step=1.0,  help="mg/dm³ — bound + free SO₂")

st.sidebar.markdown('<p class="sidebar-section">Physical Properties</p>', unsafe_allow_html=True)
density          = st.sidebar.slider("Density",             min_value=0.990, max_value=1.004,value=0.9967,step=0.0001,format="%.4f", help="g/cm³ — close to water (≈0.997)")
pH               = st.sidebar.slider("pH",                  min_value=2.74, max_value=4.01, value=3.31,  step=0.01, help="Acidity scale (lower = more acidic)")

st.sidebar.markdown('<p class="sidebar-section">Additives & Alcohol</p>', unsafe_allow_html=True)
sulphates        = st.sidebar.slider("Sulphates",           min_value=0.33, max_value=2.0,  value=0.65,  step=0.01, help="g/dm³ — wine additive (SO₂ precursor)")
alcohol          = st.sidebar.slider("Alcohol",             min_value=8.4,  max_value=14.9, value=10.4,  step=0.1,  help="% by volume")

# ── Feature summary table in main panel ───────────────────────────────────────
st.markdown("### 📋 Current Feature Values")
feature_names = [
    "Fixed Acidity", "Volatile Acidity", "Citric Acid",
    "Residual Sugar", "Chlorides", "Free Sulfur Dioxide",
    "Total Sulfur Dioxide", "Density", "pH", "Sulphates", "Alcohol"
]
feature_values = [
    fixed_acidity, volatile_acidity, citric_acid,
    residual_sugar, chlorides, free_sulfur_dio,
    total_sulfur_dio, density, pH, sulphates, alcohol
]

# Display as a neat two-column grid of metrics
col_pairs = list(zip(feature_names, feature_values))
for i in range(0, len(col_pairs), 3):
    chunk = col_pairs[i:i+3]
    cols  = st.columns(len(chunk))
    for col, (name, val) in zip(cols, chunk):
        col.metric(label=name, value=f"{val:.4f}".rstrip("0").rstrip("."))

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🍷 Predict Wine Quality")

if predict_clicked:
    # Assemble input vector in the exact column order used during training
    input_data = np.array([[
        fixed_acidity, volatile_acidity, citric_acid,
        residual_sugar, chlorides, free_sulfur_dio,
        total_sulfur_dio, density, pH, sulphates, alcohol
    ]])

    # Scale using the saved scaler
    input_scaled = scaler.transform(input_data)

    # Predict quality score
    predicted_score = int(model.predict(input_scaled)[0])

    # Get class probabilities for all known classes
    proba        = model.predict_proba(input_scaled)[0]
    classes      = model.classes_
    proba_dict   = dict(zip(classes, proba))

    # ── Qualitative interpretation ─────────────────────────────────────────────
    if predicted_score <= 4:
        card_class = "result-bad"
        emoji      = "😕"
        label      = "Bad / Low Quality Wine"
        desc       = "Needs improvement — significant flaws detected in the chemical profile."
        score_color= "#ff6b6b"
    elif predicted_score <= 6:
        card_class = "result-average"
        emoji      = "🙂"
        label      = "Average Wine"
        desc       = "Standard table wine — acceptable quality for everyday enjoyment."
        score_color= "#ffd460"
    else:
        card_class = "result-good"
        emoji      = "🌟"
        label      = "Good / High Quality Wine"
        desc       = "Premium quality — excellent chemical balance and complexity."
        score_color= "#2ecc71"

    # ── Result card ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-score" style="color:{score_color};">{emoji} {predicted_score}</div>
        <div class="result-label" style="color:{score_color};">{label}</div>
        <div class="result-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability breakdown ─────────────────────────────────────────────────
    st.markdown("#### 📊 Prediction Confidence by Quality Score")

    bar_cols = st.columns(len(classes))
    for col, cls in zip(bar_cols, sorted(classes)):
        pct = proba_dict.get(cls, 0.0) * 100
        col.metric(
            label=f"Score {cls}",
            value=f"{pct:.1f}%",
            delta=None
        )
        col.progress(int(pct))

    # ── Quality scale legend ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🏷️ Quality Scale Reference")
    legend_col1, legend_col2, legend_col3 = st.columns(3)
    with legend_col1:
        st.markdown("🔴 **Score 3–4**\nBad / Low Quality")
    with legend_col2:
        st.markdown("🟡 **Score 5–6**\nAverage Quality")
    with legend_col3:
        st.markdown("🟢 **Score 7–8**\nGood / Premium")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(255,255,255,0.1); margin-top:3rem;">
<p style="text-align:center; color:rgba(255,255,255,0.3); font-size:0.82rem;">
    Wine Quality Predictor · Trained on the UCI Wine Quality Dataset (Red Wine) ·
    Model: RandomForestClassifier
</p>
""", unsafe_allow_html=True)
