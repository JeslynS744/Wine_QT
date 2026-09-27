"""
train.py — Wine Quality Classification Model Training Script
=============================================================
Loads WineQT.csv, trains a RandomForestClassifier, evaluates it,
and saves the model + scaler to disk using joblib.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ── 1. Load the dataset ────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("WineQT.csv")

# Drop the 'Id' column — it's just a row identifier, not a feature
df.drop(columns=["Id"], inplace=True, errors="ignore")

print(f"Dataset shape after dropping 'Id': {df.shape}")
print(f"Target class distribution:\n{df['quality'].value_counts().sort_index()}\n")

# ── 2. Split into features (X) and target (y) ─────────────────────────────────
X = df.drop(columns=["quality"])
y = df["quality"]

feature_names = X.columns.tolist()
print(f"Features ({len(feature_names)}): {feature_names}\n")

# ── 3. Train / test split — 80% train, 20% test ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # preserve class proportions in both splits
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")

# ── 4. Feature scaling ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on training data only
X_test_scaled  = scaler.transform(X_test)         # apply same transform to test

# ── 5. Train Random Forest Classifier ─────────────────────────────────────────
print("Training RandomForestClassifier …")
model = RandomForestClassifier(
    n_estimators=200,       # 200 trees for a good bias/variance trade-off
    max_depth=None,         # let trees grow fully
    min_samples_split=4,    # require at least 4 samples to split a node
    min_samples_leaf=2,     # require at least 2 samples at each leaf
    class_weight="balanced",# compensate for imbalanced quality classes
    random_state=42,
    n_jobs=-1               # use all available CPU cores
)
model.fit(X_train_scaled, y_train)
print("Training complete.\n")

# ── 6. Evaluate the model ─────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Feature importances (top 5)
importances = pd.Series(model.feature_importances_, index=feature_names)
print("Top 5 Feature Importances:")
print(importances.sort_values(ascending=False).head(5).to_string())
print()

# ── 7. Save the model and scaler to disk ─────────────────────────────────────
joblib.dump(model,  "wine_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Saved: wine_model.pkl")
print("Saved: scaler.pkl")
print("\nAll done! You can now launch the Streamlit app with: streamlit run app.py")
