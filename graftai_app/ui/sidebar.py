# ═══════════════════════════════════════════════════════
#  ui/sidebar.py — Barre latérale Streamlit
#  Retourne : infos patient + paramètres XAI
# ═══════════════════════════════════════════════════════

import streamlit as st
from sklearn.metrics import f1_score
from config import DEFAULT_LIME_SAMPLES, DEFAULT_LIME_FEATURES


def render_sidebar(model, X_train, y_train) -> dict:
    """
    Affiche la barre latérale et retourne un dict de paramètres :

    {
        "patient_nom"      : str,
        "patient_id"       : str,
        "medecin"          : str,
        "n_lime_samples"   : int,
        "n_lime_features"  : int,
        "show_force_plot"  : bool,
        "show_distribution": bool,
    }
    """
    with st.sidebar:

        # ── Identification ─────────────────────────────
        st.markdown("##  Identification")
        patient_nom = st.text_input("Nom du patient",   value="Dupont Jean",    key="nom")
        patient_id  = st.text_input("ID Dossier",       value="TRP-2024-0847",  key="id")
        medecin     = st.text_input("Médecin référent", value="Dr. Martin",     key="med")

        st.divider()

        # ── Paramètres XAI ─────────────────────────────
        st.markdown("##  Paramètres XAI")
        n_lime_samples = st.slider(
            "Échantillons LIME",
            min_value=200, max_value=1000,
            value=DEFAULT_LIME_SAMPLES, step=50,
        )
        n_lime_features = st.slider(
            "Features LIME actives",
            min_value=6, max_value=18,
            value=DEFAULT_LIME_FEATURES,
        )
        show_force_plot   = st.checkbox("Afficher SHAP Force Plot",      value=True)
        show_distribution = st.checkbox("Afficher distribution population", value=True)

        st.divider()

        # ── Infos modèle ────────────────────────────────
        st.markdown("##  Modèle")
        train_acc = (model.predict(X_train) == y_train).mean()
        train_f1 = f1_score(y_train, model.predict(X_train))
        st.metric("F1 Score (train)", f"{train_f1:.1%}")
        st.metric("Précision (train)", f"{train_acc:.1%}")
        st.metric("Features totales",  str(X_train.shape[1]))
        st.metric("Patients référence", f"{len(X_train):,}")

    return {
        "patient_nom":       patient_nom,
        "patient_id":        patient_id,
        "medecin":           medecin,
        "n_lime_samples":    n_lime_samples,
        "n_lime_features":   n_lime_features,
        "show_force_plot":   show_force_plot,
        "show_distribution": show_distribution,
    }