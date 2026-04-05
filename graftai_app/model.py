# ═══════════════════════════════════════════════════════
#  model.py — Chargement du vrai modèle GraftAI
# ═══════════════════════════════════════════════════════
import numpy as np
import joblib
import streamlit as st
from config import ALL_FEATURE_NAMES

@st.cache_resource(show_spinner="⚙️ Chargement du modèle ML...")
def load_model():
    """Charge model + X_train + y_train depuis les .pkl"""
    model   = joblib.load("model.pkl")
    transformer = joblib.load("transformer.pkl")  # ✅ AJOUTÉ 
    X_train = joblib.load("X_train.pkl")
    y_train = joblib.load("y_train.pkl")
    
    return model, transformer, X_train, y_train, ALL_FEATURE_NAMES

def predict(model, input_vec: np.ndarray) -> float:
    """Retourne la probabilité de rejet (classe 1)."""
    return float(model.predict_proba(input_vec.reshape(1, -1))[0, 1])

def get_train_probabilities(model, X_train: np.ndarray) -> np.ndarray:
    """Retourne les probabilités de rejet sur le jeu d'entraînement."""
    return model.predict_proba(X_train)[:, 1]