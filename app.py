"""
app.py — Wine Quality Predictor — Streamlit Web App
=====================================================
A modern, dark-mode, Figma-inspired physicochemical evaluation dashboard
for predicting red wine quality using RandomForest classification.
"""

import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Robust base directory — works locally AND on cloud deployments
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()

# ── 1. Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 2. Custom CSS: Figma Dark-Mode Aesthetic & Sidebar Removal ─────────────────
st.markdown("""
<style>
/* ── Typography & Base Styles ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #f1f5f9;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    overflow-x: hidden !important;
}

/* Background canvas */
.stApp {
    background: radial-gradient(circle at 50% 0%, #200b17 0%, #110710 50%, #0a0409 100%) !important;
    min-height: 100vh;
}

/* ── Eliminate Default Left Sidebar Completely & Remove Dividers ── */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
[data-testid="collapsedControl"],
div[data-testid="stSidebarCollapsedControl"],
button[data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stAppViewContainer"] {
    margin-left: 0 !important;
    padding-left: 0 !important;
}

/* Hide standard Streamlit header chrome */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
header[data-testid="stHeader"] {
    display: none !important;
}

/* Block container padding & max width */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1360px !important;
    margin: 0 auto !important;
}

/* ── Top Header Banner (Edge-to-Edge) ── */
.top-header-banner {
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    margin-right: calc(-50vw + 50%);
    padding: 2.6rem 1.5rem 2.2rem;
    background: linear-gradient(180deg, rgba(38, 12, 26, 0.95) 0%, rgba(20, 7, 16, 0.98) 75%, rgba(13, 5, 11, 0.95) 100%);
    border-bottom: 1px solid rgba(244, 63, 94, 0.28);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.65), inset 0 -1px 0 rgba(255, 255, 255, 0.05);
    text-align: center;
    position: relative;
    top: 0;
    margin-top: -1rem;
    margin-bottom: 2.2rem;
}

.banner-content {
    max-width: 900px;
    margin: 0 auto;
}

.header-logo-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 68px;
    height: 68px;
    background: radial-gradient(circle, rgba(244, 63, 94, 0.25) 0%, rgba(159, 18, 57, 0.1) 80%);
    border: 1px solid rgba(244, 63, 94, 0.4);
    border-radius: 20px;
    box-shadow: 0 0 24px rgba(225, 29, 72, 0.35);
    margin-bottom: 0.8rem;
    font-size: 2.3rem;
    animation: gentle-glow 3s ease-in-out infinite alternate;
}

@keyframes gentle-glow {
    0% { box-shadow: 0 0 16px rgba(225, 29, 72, 0.25); transform: translateY(0); }
    100% { box-shadow: 0 0 28px rgba(244, 63, 94, 0.45); transform: translateY(-2px); }
}

.header-title {
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(90deg, #ff758c 0%, #ff8e53 50%, #fecdd3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.2rem 0 0.5rem;
    line-height: 1.15;
}

.header-subtitle {
    color: rgba(241, 245, 249, 0.72);
    font-size: 1.05rem;
    font-weight: 400;
    letter-spacing: 0.01em;
    margin: 0;
}

/* ── Categorized Input Cards ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(24, 12, 22, 0.72) !important;
    border: 1px solid rgba(244, 63, 94, 0.18) !important;
    border-radius: 16px !important;
    padding: 1.25rem 1.4rem !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.38) !important;
    backdrop-filter: blur(12px) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(244, 63, 94, 0.4) !important;
    box-shadow: 0 12px 36px rgba(225, 29, 72, 0.18) !important;
    transform: translateY(-2px);
}

.card-category-header {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.85rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.card-category-icon {
    font-size: 1.35rem;
    line-height: 1;
}

.card-category-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #ffe4e6;
    letter-spacing: 0.01em;
}

.card-category-badge {
    margin-left: auto;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    background: rgba(244, 63, 94, 0.14);
    color: #fda4af;
    border: 1px solid rgba(244, 63, 94, 0.25);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Slider aesthetic overrides */
div[data-testid="stSlider"] label {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    margin-bottom: 0.2rem !important;
}

div[data-testid="stSlider"] [data-baseweb="slider"] {
    margin-top: 0.3rem !important;
}

div[data-testid="stSlider"] div[role="slider"] {
    background-color: #f43f5e !important;
    border: 2px solid #ffffff !important;
    box-shadow: 0 0 10px rgba(244, 63, 94, 0.7) !important;
}

/* ── Interactive Feature Values Panel ── */
.values-panel-card {
    background: rgba(20, 10, 18, 0.8) !important;
    border: 1px solid rgba(244, 63, 94, 0.22) !important;
    border-radius: 18px !important;
    padding: 1.5rem 1.75rem 1.25rem !important;
    box-shadow: 0 10px 32px rgba(0, 0, 0, 0.45) !important;
    backdrop-filter: blur(14px) !important;
    margin-top: 1.8rem;
    margin-bottom: 1.5rem;
}

.values-header-flex {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.2rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.values-title {
    font-size: 1.32rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.values-subtitle {
    font-size: 0.9rem;
    color: rgba(241, 245, 249, 0.65);
    margin: 0.25rem 0 0;
}

.sync-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.85rem;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #6ee7b7;
    letter-spacing: 0.02em;
}

.sync-dot {
    width: 7px;
    height: 7px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
    animation: sync-pulse 1.8s infinite;
}

@keyframes sync-pulse {
    0% { transform: scale(0.9); opacity: 0.6; }
    50% { transform: scale(1.3); opacity: 1; }
    100% { transform: scale(0.9); opacity: 0.6; }
}

/* Number Inputs styling */
div[data-testid="stNumberInput"] label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

div[data-testid="stNumberInput"] input {
    background: rgba(14, 7, 13, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 9px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.92rem !important;
    padding: 0.45rem 0.65rem !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stNumberInput"] input:focus {
    border-color: #f43f5e !important;
    box-shadow: 0 0 0 3px rgba(244, 63, 94, 0.22) !important;
    background: rgba(22, 9, 19, 0.95) !important;
}

div[data-testid="stNumberInput"] button {
    background: rgba(30, 14, 25, 0.85) !important;
    color: #fda4af !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

div[data-testid="stNumberInput"] button:hover {
    background: rgba(225, 29, 72, 0.3) !important;
    color: #ffffff !important;
}

/* ── Predict Button ── */
div.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #9f1239 0%, #be123c 45%, #e11d48 100%) !important;
    color: #ffffff !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    padding: 0.95rem 2rem !important;
    border: none !important;
    border-radius: 14px !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 10px 28px rgba(225, 29, 72, 0.42) !important;
    letter-spacing: 0.02em !important;
    margin-top: 0.5rem !important;
    margin-bottom: 1.5rem !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 38px rgba(244, 63, 94, 0.58) !important;
    background: linear-gradient(135deg, #881337 0%, #be123c 40%, #f43f5e 100%) !important;
}

div.stButton > button:active {
    transform: translateY(1px) !important;
}

/* ── Prediction Result Cards ── */
.result-card-container {
    padding: 2.2rem 2.5rem;
    border-radius: 20px;
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 2rem;
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.55);
    position: relative;
    overflow: hidden;
}

.result-bad {
    background: linear-gradient(135deg, rgba(76, 14, 25, 0.85) 0%, rgba(45, 10, 16, 0.92) 100%);
    border: 2px solid rgba(239, 68, 68, 0.6);
}

.result-average {
    background: linear-gradient(135deg, rgba(82, 46, 11, 0.85) 0%, rgba(48, 26, 8, 0.92) 100%);
    border: 2px solid rgba(245, 158, 11, 0.6);
}

.result-good {
    background: linear-gradient(135deg, rgba(13, 66, 39, 0.85) 0%, rgba(7, 40, 24, 0.92) 100%);
    border: 2px solid rgba(16, 185, 129, 0.6);
}

.result-score-digit {
    font-size: 5.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.4rem;
    letter-spacing: -0.04em;
    font-family: 'Inter', sans-serif;
}

.result-label-badge {
    display: inline-block;
    font-size: 1.45rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
    letter-spacing: -0.01em;
}

.result-desc-text {
    font-size: 1.05rem;
    color: rgba(241, 245, 249, 0.85);
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Stat pills */
.stat-pill {
    background: rgba(28, 14, 25, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.stat-pill-label {
    font-size: 0.78rem;
    color: rgba(241, 245, 249, 0.6);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}

.stat-pill-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
}

.empty-state-card {
    text-align: center;
    padding: 2.5rem 1.5rem;
    background: rgba(24, 12, 22, 0.5);
    border: 1px dashed rgba(244, 63, 94, 0.25);
    border-radius: 16px;
    margin: 1rem 0;
    color: rgba(241, 245, 249, 0.65);
}
</style>
""", unsafe_allow_html=True)

# ── 3. Model Auto-train & Artifact Loader ───────────────────────────────────────
def auto_train():
    """Train and save model + scaler if not present on disk."""
    model_path  = os.path.join(BASE_DIR, "wine_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return

    csv_path = os.path.join(BASE_DIR, "WineQT.csv")
    if not os.path.exists(csv_path):
        st.error(f"WineQT.csv not found in {BASE_DIR}. Cannot train model.")
        return

    with st.spinner("Initializing first-time model training... (~5 seconds)"):
        df = pd.read_csv(csv_path)
        df.drop(columns=["Id"], inplace=True, errors="ignore")
        X = df.drop(columns=["quality"])
        y = df["quality"]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(
            n_estimators=200, min_samples_split=4, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        joblib.dump(model,  model_path)
        joblib.dump(scaler, scaler_path)

    st.cache_resource.clear()
    st.rerun()

auto_train()

@st.cache_resource
def load_artifacts():
    """Load and cache the trained model and scaler."""
    model_path  = os.path.join(BASE_DIR, "wine_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.error("⚠️ Model artifacts could not be loaded. Please ensure wine_model.pkl and scaler.pkl exist.")
    st.stop()

# ── 4. Feature Specifications & Metadata ───────────────────────────────────────
FEATURE_CONFIG = {
    "fixed_acidity": {
        "col": "fixed acidity",
        "name": "Fixed Acidity",
        "unit": "g/dm³",
        "min": 4.6, "max": 15.9, "default": 8.3, "step": 0.1, "format": "%.1f",
        "help": "Tartaric acid equivalents (g/dm³) — gives structure to the wine."
    },
    "volatile_acidity": {
        "col": "volatile acidity",
        "name": "Volatile Acidity",
        "unit": "g/dm³",
        "min": 0.12, "max": 1.58, "default": 0.53, "step": 0.01, "format": "%.2f",
        "help": "Acetic acid (g/dm³) — excessive levels impart an unpleasantly sharp vinegar taste."
    },
    "citric_acid": {
        "col": "citric acid",
        "name": "Citric Acid",
        "unit": "g/dm³",
        "min": 0.00, "max": 1.00, "default": 0.27, "step": 0.01, "format": "%.2f",
        "help": "Adds freshness, crispness, and subtle citrus aromas (g/dm³)."
    },
    "residual_sugar": {
        "col": "residual sugar",
        "name": "Residual Sugar",
        "unit": "g/dm³",
        "min": 0.9, "max": 15.5, "default": 2.5, "step": 0.1, "format": "%.1f",
        "help": "Unfermented natural grape sugars remaining after fermentation (g/dm³)."
    },
    "chlorides": {
        "col": "chlorides",
        "name": "Chlorides",
        "unit": "g/dm³",
        "min": 0.012, "max": 0.611, "default": 0.087, "step": 0.001, "format": "%.3f",
        "help": "Total sodium chloride / salt content in the wine (g/dm³)."
    },
    "free_sulfur_dio": {
        "col": "free sulfur dioxide",
        "name": "Free Sulfur Dioxide",
        "unit": "mg/dm³",
        "min": 1.0, "max": 68.0, "default": 15.6, "step": 0.5, "format": "%.1f",
        "help": "Unbound active SO₂ (mg/dm³) — prevents microbial oxidation."
    },
    "total_sulfur_dio": {
        "col": "total sulfur dioxide",
        "name": "Total Sulfur Dioxide",
        "unit": "mg/dm³",
        "min": 6.0, "max": 289.0, "default": 45.9, "step": 1.0, "format": "%.1f",
        "help": "Total bound and free SO₂ (mg/dm³) — preserves freshness and protects aging."
    },
    "density": {
        "col": "density",
        "name": "Density",
        "unit": "g/cm³",
        "min": 0.9900, "max": 1.0040, "default": 0.9967, "step": 0.0001, "format": "%.4f",
        "help": "Liquid density (g/cm³) influenced by alcohol content and dissolved sugar."
    },
    "pH": {
        "col": "pH",
        "name": "pH",
        "unit": "",
        "min": 2.74, "max": 4.01, "default": 3.31, "step": 0.01, "format": "%.2f",
        "help": "Scale of wine acidity (typically 3.0–3.8; lower represents higher acidity)."
    },
    "sulphates": {
        "col": "sulphates",
        "name": "Sulphates",
        "unit": "g/dm³",
        "min": 0.33, "max": 2.00, "default": 0.65, "step": 0.01, "format": "%.2f",
        "help": "Potassium sulphate additive (g/dm³) promoting antibacterial & antioxidant protection."
    },
    "alcohol": {
        "col": "alcohol",
        "name": "Alcohol",
        "unit": "% vol",
        "min": 8.4, "max": 14.9, "default": 10.4, "step": 0.1, "format": "%.1f",
        "help": "Percent ethanol by volume (% vol) contributing to body and mouthfeel."
    },
}

# ── 5. Session State Synchronization Handlers ──────────────────────────────────
# Initialize session state keys once
for key, cfg in FEATURE_CONFIG.items():
    if f"slider_{key}" not in st.session_state:
        st.session_state[f"slider_{key}"] = cfg["default"]
    if f"num_{key}" not in st.session_state:
        st.session_state[f"num_{key}"] = cfg["default"]

def on_slider_change(k):
    """Callback when a slider moves: sync to numeric input."""
    st.session_state[f"num_{k}"] = st.session_state[f"slider_{k}"]

def on_num_change(k):
    """Callback when a number input is typed: clamp, round, and sync to slider."""
    val = st.session_state[f"num_{k}"]
    cfg = FEATURE_CONFIG[k]
    clamped = max(cfg["min"], min(cfg["max"], val))
    step = cfg["step"]
    if step == 0.0001:
        clamped = round(clamped, 4)
    elif step == 0.001:
        clamped = round(clamped, 3)
    elif step == 0.01:
        clamped = round(clamped, 2)
    elif step in (0.1, 0.5):
        clamped = round(clamped, 1)
    else:
        clamped = round(clamped, 1)
    st.session_state[f"slider_{k}"] = clamped
    st.session_state[f"num_{k}"] = clamped

def render_slider(key):
    """Helper to render a slider linked to session state."""
    cfg = FEATURE_CONFIG[key]
    unit_str = f" ({cfg['unit']})" if cfg['unit'] else ""
    st.slider(
        label=f"{cfg['name']}{unit_str}",
        min_value=cfg["min"],
        max_value=cfg["max"],
        step=cfg["step"],
        format=cfg["format"],
        key=f"slider_{key}",
        on_change=on_slider_change,
        args=(key,),
        help=cfg["help"]
    )

# ── 6. Section 1: Top Header Banner (Edge-to-Edge) ─────────────────────────────
st.markdown("""
<div class="top-header-banner">
    <div class="banner-content">
        <div class="header-logo-badge">🍷</div>
        <h1 class="header-title">Wine Quality Predictor</h1>
        <p class="header-subtitle">
            Physicochemical Analysis & Machine Learning Classification Platform
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 7. Section 2: Categorized Input Cards (Top Section: 3 in Row 1, 2 in Row 2) ─
st.markdown("""
<div style="margin-bottom: 0.9rem;">
    <span style="font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #fda4af;">
        Step 1 — Physicochemical Parameters
    </span>
    <h3 style="margin: 0.15rem 0 0; font-size: 1.35rem; font-weight: 700; color: #ffffff;">
        Categorized Laboratory Measurements
    </h3>
</div>
""", unsafe_allow_html=True)

# Row 1: 3 cards (Acidity, Sugar & Salt, Sulfur Dioxide)
r1_col1, r1_col2, r1_col3 = st.columns(3, gap="medium")

with r1_col1:
    with st.container(border=True):
        st.markdown("""
        <div class="card-category-header">
            <span class="card-category-icon">🧪</span>
            <span class="card-category-title">Acidity</span>
            <span class="card-category-badge">3 Features</span>
        </div>
        """, unsafe_allow_html=True)
        render_slider("fixed_acidity")
        render_slider("volatile_acidity")
        render_slider("citric_acid")

with r1_col2:
    with st.container(border=True):
        st.markdown("""
        <div class="card-category-header">
            <span class="card-category-icon">🍬</span>
            <span class="card-category-title">Sugar & Salt</span>
            <span class="card-category-badge">2 Features</span>
        </div>
        """, unsafe_allow_html=True)
        render_slider("residual_sugar")
        render_slider("chlorides")

with r1_col3:
    with st.container(border=True):
        st.markdown("""
        <div class="card-category-header">
            <span class="card-category-icon">💨</span>
            <span class="card-category-title">Sulfur Dioxide</span>
            <span class="card-category-badge">2 Features</span>
        </div>
        """, unsafe_allow_html=True)
        render_slider("free_sulfur_dio")
        render_slider("total_sulfur_dio")

# Row 2: 2 cards (Physical Properties, Additives and Alcohol)
r2_col1, r2_col2 = st.columns(2, gap="medium")

with r2_col1:
    with st.container(border=True):
        st.markdown("""
        <div class="card-category-header">
            <span class="card-category-icon">⚖️</span>
            <span class="card-category-title">Physical Properties</span>
            <span class="card-category-badge">2 Features</span>
        </div>
        """, unsafe_allow_html=True)
        render_slider("density")
        render_slider("pH")

with r2_col2:
    with st.container(border=True):
        st.markdown("""
        <div class="card-category-header">
            <span class="card-category-icon">🍷</span>
            <span class="card-category-title">Additives and Alcohol</span>
            <span class="card-category-badge">2 Features</span>
        </div>
        """, unsafe_allow_html=True)
        render_slider("sulphates")
        render_slider("alcohol")

# ── 8. Section 3: Interactive "Current Feature Values" (Middle Section) ────────
st.markdown("""
<div class="values-panel-card">
    <div class="values-header-flex">
        <div>
            <h3 class="values-title">📋 Current Feature Values</h3>
            <p class="values-subtitle">
                Interactive values synced in real-time with the sliders above. Click or type to calibrate precise values.
            </p>
        </div>
        <div class="sync-badge">
            <span class="sync-dot"></span> Bidirectional Sync Active
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Grid layout for the 11 feature number inputs
# Row A: 4 inputs
row_a_keys = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar"]
cols_a = st.columns(4, gap="medium")
for col, key in zip(cols_a, row_a_keys):
    cfg = FEATURE_CONFIG[key]
    unit_str = f" ({cfg['unit']})" if cfg['unit'] else ""
    with col:
        st.number_input(
            label=f"{cfg['name']}{unit_str}",
            min_value=cfg["min"],
            max_value=cfg["max"],
            step=cfg["step"],
            format=cfg["format"],
            key=f"num_{key}",
            on_change=on_num_change,
            args=(key,),
            help=f"Range: {cfg['min']} - {cfg['max']} {cfg['unit']}"
        )

# Row B: 4 inputs
row_b_keys = ["chlorides", "free_sulfur_dio", "total_sulfur_dio", "density"]
cols_b = st.columns(4, gap="medium")
for col, key in zip(cols_b, row_b_keys):
    cfg = FEATURE_CONFIG[key]
    unit_str = f" ({cfg['unit']})" if cfg['unit'] else ""
    with col:
        st.number_input(
            label=f"{cfg['name']}{unit_str}",
            min_value=cfg["min"],
            max_value=cfg["max"],
            step=cfg["step"],
            format=cfg["format"],
            key=f"num_{key}",
            on_change=on_num_change,
            args=(key,),
            help=f"Range: {cfg['min']} - {cfg['max']} {cfg['unit']}"
        )

# Row C: 3 inputs
row_c_keys = ["pH", "sulphates", "alcohol"]
cols_c = st.columns(3, gap="medium")
for col, key in zip(cols_c, row_c_keys):
    cfg = FEATURE_CONFIG[key]
    unit_str = f" ({cfg['unit']})" if cfg['unit'] else ""
    with col:
        st.number_input(
            label=f"{cfg['name']}{unit_str}",
            min_value=cfg["min"],
            max_value=cfg["max"],
            step=cfg["step"],
            format=cfg["format"],
            key=f"num_{key}",
            on_change=on_num_change,
            args=(key,),
            help=f"Range: {cfg['min']} - {cfg['max']} {cfg['unit']}"
        )

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ── 9. Section 4: Prediction Action & Results (Bottom Section) ─────────────────
predict_clicked = st.button("🍷 Predict Wine Quality", use_container_width=True)

feature_keys_in_order = [
    "fixed_acidity", "volatile_acidity", "citric_acid",
    "residual_sugar", "chlorides", "free_sulfur_dio",
    "total_sulfur_dio", "density", "pH", "sulphates", "alcohol"
]

feature_columns_in_order = [FEATURE_CONFIG[k]["col"] for k in feature_keys_in_order]
current_feature_values = [st.session_state[f"slider_{k}"] for k in feature_keys_in_order]

if predict_clicked:
    # Use DataFrame with exact trained column names to prevent feature-name warnings
    input_df = pd.DataFrame([current_feature_values], columns=feature_columns_in_order)
    input_scaled = scaler.transform(input_df)

    predicted_score = int(model.predict(input_scaled)[0])
    proba = model.predict_proba(input_scaled)[0]
    classes = model.classes_
    proba_dict = dict(zip(classes, proba))

    st.session_state["prediction_output"] = {
        "score": predicted_score,
        "proba_dict": proba_dict,
        "classes": classes,
        "confidence": proba_dict.get(predicted_score, 0.0) * 100,
        "input_data": current_feature_values
    }

if "prediction_output" in st.session_state:
    res = st.session_state["prediction_output"]
    score = res["score"]
    proba_dict = res["proba_dict"]
    classes = res["classes"]
    confidence = res["confidence"]

    # Qualitative classification
    if score <= 4:
        card_class = "result-bad"
        emoji = "😕"
        label = "Bad / Low Quality Wine"
        desc = "Significant chemical flaws or imbalance detected. Acidity or volatile compounds are outside optimal vintage thresholds."
        score_color = "#f87171"
    elif score <= 6:
        card_class = "result-average"
        emoji = "🙂"
        label = "Average Table Wine"
        desc = "Standard commercial table wine profile with satisfactory balance. Suitable for everyday dining."
        score_color = "#fbbf24"
    else:
        card_class = "result-good"
        emoji = "🌟"
        label = "Good / High Quality Wine"
        desc = "Exceptional chemical balance, complexity, and alcohol integration. Characteristic of reserve or premium vintage quality."
        score_color = "#34d399"

    # Main evaluation banner
    st.markdown(f"""
    <div class="result-card-container {card_class}">
        <div class="result-score-digit" style="color:{score_color};">{emoji} {score}</div>
        <div class="result-label-badge" style="color:{score_color};">{label}</div>
        <p class="result-desc-text">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Evaluation Statistics
    st.markdown("#### 📈 Key Physicochemical Diagnostics & Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4, gap="medium")

    # Acidity balance ratio
    fixed_val = st.session_state["slider_fixed_acidity"]
    vol_val = max(st.session_state["slider_volatile_acidity"], 0.01)
    acidity_ratio = fixed_val / vol_val

    # SO2 ratio
    free_so2 = st.session_state["slider_free_sulfur_dio"]
    tot_so2 = max(st.session_state["slider_total_sulfur_dio"], 1.0)
    so2_ratio = (free_so2 / tot_so2) * 100

    alc_val = st.session_state["slider_alcohol"]

    with stat_col1:
        st.markdown(f"""
        <div class="stat-pill">
            <div class="stat-pill-label">Model Confidence</div>
            <div class="stat-pill-value" style="color:{score_color};">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with stat_col2:
        st.markdown(f"""
        <div class="stat-pill">
            <div class="stat-pill-label">Fixed / Volatile Acidity</div>
            <div class="stat-pill-value">{acidity_ratio:.1f}x</div>
        </div>
        """, unsafe_allow_html=True)

    with stat_col3:
        st.markdown(f"""
        <div class="stat-pill">
            <div class="stat-pill-label">Active Free SO₂ Ratio</div>
            <div class="stat-pill-value">{so2_ratio:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with stat_col4:
        st.markdown(f"""
        <div class="stat-pill">
            <div class="stat-pill-label">Alcohol Strength</div>
            <div class="stat-pill-value">{alc_val:.1f}% vol</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Prediction Confidence Distribution
    st.markdown("#### 📊 Prediction Confidence by Quality Score")
    bar_cols = st.columns(len(classes))
    for col, cls in zip(bar_cols, sorted(classes)):
        pct = proba_dict.get(cls, 0.0) * 100
        is_winner = (cls == score)
        border_accent = f"border: 1px solid {score_color};" if is_winner else ""
        col.metric(
            label=f"Score {cls}" + (" (Predicted)" if is_winner else ""),
            value=f"{pct:.1f}%"
        )
        col.progress(int(min(max(pct, 0.0), 100.0)))

    # Quality Scale Reference
    st.markdown("---")
    st.markdown("#### 🏷️ Quality Scale Reference")
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        st.markdown("""
        <div style="background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 10px; padding: 0.9rem; text-align: center;">
            <b style="color: #f87171;">🔴 Score 3–4</b><br>
            <span style="font-size: 0.85rem; color: #cbd5e1;">Bad / Low Quality (Defective balance)</span>
        </div>
        """, unsafe_allow_html=True)
    with lcol2:
        st.markdown("""
        <div style="background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); border-radius: 10px; padding: 0.9rem; text-align: center;">
            <b style="color: #fbbf24;">🟡 Score 5–6</b><br>
            <span style="font-size: 0.85rem; color: #cbd5e1;">Average Quality (Daily table wine)</span>
        </div>
        """, unsafe_allow_html=True)
    with lcol3:
        st.markdown("""
        <div style="background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 10px; padding: 0.9rem; text-align: center;">
            <b style="color: #34d399;">🟢 Score 7–8</b><br>
            <span style="font-size: 0.85rem; color: #cbd5e1;">Good / Premium (Reserve grade)</span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state-card">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🍇</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #ffffff; margin-bottom: 0.3rem;">
            Ready to Evaluate Quality
        </div>
        <p style="margin: 0; font-size: 0.9rem;">
            Adjust the sliders or type exact values in the panels above, then click <b>Predict Wine Quality</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── 10. Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.08); text-align: center;">
    <p style="color: rgba(241, 245, 249, 0.4); font-size: 0.85rem; margin: 0;">
        Wine Quality Predictor · Trained on the UCI Wine Quality Dataset (Red Wine) ·
        Model: RandomForestClassifier
    </p>
</div>
""", unsafe_allow_html=True)
