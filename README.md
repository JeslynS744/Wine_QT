# Wine Quality Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pandas](https://img.shields.io/badge/pandas-2.0+-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

**Wine Quality Prediction** is an end-to-end machine learning web application that classifies and predicts red wine quality scores based on 11 physicochemical properties using a Random Forest classification pipeline and an interactive Streamlit interface.

---

## 🌟 Key Features

* **Automated Data Ingestion & Preprocessing:** Ingests red wine samples from the UCI Wine Quality dataset (`WineQT.csv`), removes non-predictive identifiers, and profiles distributions across 11 physicochemical properties.
* **Stratified Splitting & Scaling Pipeline:** Executes an 80/20 train-test split stratified across wine quality ratings to preserve class balance, coupled with `StandardScaler` feature normalization.
* **Balanced Random Forest Classification:** Deploys an ensemble of 200 decision trees with balanced class weighting (`class_weight="balanced"`) to counteract label skew and maximize multi-class prediction accuracy.
* **Physicochemical Feature Importance:** Evaluates and ranks key chemical drivers of wine quality—identifying alcohol content, volatile acidity, and sulphates as top predictive factors.
* **Interactive Streamlit Web Interface:** Delivers a responsive dashboard with real-time sliders for parameter tuning, immediate quality score predictions (3–8), and categorical tier badges (Bad, Average, Good).
* **Cold-Start Artifact Persistence:** Automatically trains and serializes model and scaler artifacts (`wine_model.pkl`, `scaler.pkl`) on first startup if missing from disk.

---

## 🚀 Quick Start

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/JeslynS744/Wine_QT.git
cd Wine_QT
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1 | Mac/Linux: source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
```bash
python train.py
```
> *Note: Generates `wine_model.pkl` and `scaler.pkl`. If skipped, `app.py` automatically trains and saves them on first run.*

### 4. Launch the Web Application
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
Wine_QT/
├── .gitignore
├── README.md
├── WineQT.csv
├── app.py
├── requirements.txt
├── scaler.pkl
├── train.py
└── wine_model.pkl
```
