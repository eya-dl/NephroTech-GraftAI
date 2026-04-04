# ═══════════════════════════════════════════════════════
#  ui/inputs.py — Formulaire de saisie des attributs cliniques
# ═══════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


@st.cache_resource
def load_transformer():
    transformer = joblib.load("transformer.pkl")
    col_order   = joblib.load("col_order.pkl")  # ordre exact du notebook
    return transformer, col_order


def render_patient_form() -> tuple[np.ndarray, list[float], list[float]]:
    st.markdown("## Saisie des Attributs Cliniques")

    # ── Variables numériques ─────────────────────────────
    num_values: list[float] = []
    cols = st.columns(3)

    for i, (feat, cfg) in enumerate(NUMERICAL_FEATURES.items()):
        col = cols[i % 3]
        with col:
            if cfg["type"] == "int":
                val = st.number_input(feat,
                    min_value=int(cfg["min"]), max_value=int(cfg["max"]),
                    value=int(cfg["default"]), step=1, key=f"num_{i}",
                    help=f"Unité : {cfg['unit']}" if cfg["unit"] else None)
            else:
                val = st.number_input(feat,
                    min_value=float(cfg["min"]), max_value=float(cfg["max"]),
                    value=float(cfg["default"]), step=0.1, format="%.1f",
                    key=f"num_{i}",
                    help=f"Unité : {cfg['unit']}" if cfg["unit"] else None)
        num_values.append(float(val))

    st.markdown("---")
    st.markdown("### Variables Catégorielles")

    # ── Variables catégorielles ──────────────────────────
    cat_values_raw: dict = {}
    cat_values_idx: list[float] = []
    cols_cat = st.columns(3)

    for i, (feat, options) in enumerate(CATEGORICAL_FEATURES.items()):
        col = cols_cat[i % 3]
        with col:
            sel = st.selectbox(feat, options, key=f"cat_{i}")
            cat_values_raw[feat] = sel
            cat_values_idx.append(float(options.index(sel)))

    # ── Construire le DataFrame dans l'ordre exact du notebook ──
    transformer, col_order = load_transformer()

    num_dict = {feat: val for feat, val in zip(NUMERICAL_FEATURES.keys(), num_values)}
    all_vals = {**num_dict, **cat_values_raw}

    # ⚠️ Réordonner selon col_order — critique pour la compatibilité
    try:
        input_df = pd.DataFrame([[all_vals[c] for c in col_order]], columns=col_order)
    except KeyError as e:
        st.error(f"⚠️ Colonne manquante : {e}. Vérifie que config.py correspond au dataset.")
        st.stop()

    # ── Transformer ──────────────────────────────────────
    input_vec = transformer.transform(input_df)[0]

    return input_vec, num_values, cat_values_idx