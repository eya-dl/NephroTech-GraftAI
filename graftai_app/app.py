# ═══════════════════════════════════════════════════════
#  app.py — Point d'entrée GraftAI  (avec DiCE)
# ═══════════════════════════════════════════════════════
import warnings
import pandas as pd
import streamlit as st
warnings.filterwarnings("ignore")

from config import APP_TITLE, APP_ICON, ALL_FEATURE_NAMES
from model import load_model, predict
from utils.helpers import inject_css
from ui.sidebar import render_sidebar
from ui.inputs import render_patient_form
from ui.display import (
    render_header,
    render_prediction_result,
    render_xai_tabs,
    render_report_section,
)
from ui.chat_widget import render_chat_widget
from ui.dice_display import render_dice_section
from dice_runner import build_dice_explainer, load_train_raw

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Initialisation de DiCE…")
def load_dice_explainer(_model, _transformer):
    """
    Construit l'explainer DiCE une seule fois et le met en cache.
    """
    train_raw = load_train_raw("train_dataset.xlsx")
    return build_dice_explainer(_model, _transformer, train_raw)


def main():
    inject_css()
    render_header()

    # ── Chargement modèle ────────────────────────────────────────────────────
    model, transformer, X_train, y_train, feature_names = load_model()
    params = render_sidebar(model, X_train, y_train)

    # ── Formulaire patient ───────────────────────────────────────────────────
    input_vec, num_vals, cat_vals, input_df = render_patient_form()
    input_values_full = num_vals + cat_vals

    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run = st.button("🔬 Analyser & Prédire le Risque de Rejet",
                        use_container_width=True)

    # ── Prédiction + XAI + DiCE ─────────────────────────────────────────────
    if run or st.session_state.get("ran", False):
        st.session_state["ran"] = True

        prob      = predict(model, input_vec)
        fig_gauge = render_prediction_result(prob)

        xai_results = render_xai_tabs(
            model, X_train, input_vec, input_values_full,
            n_lime_samples=params["n_lime_samples"],
            n_lime_features=params["n_lime_features"],
            show_force_plot=params["show_force_plot"],
            show_distribution=params["show_distribution"],
            transformer=transformer,
        )

        render_report_section(
            patient_info={"nom": params["patient_nom"],
                          "id":  params["patient_id"],
                          "medecin": params["medecin"]},
            prob=prob, xai_results=xai_results,
            input_values_full=input_values_full,
            fig_gauge=fig_gauge, model=model,
            X_train=X_train, input_vec=input_vec,
            transformer=transformer,
        )

        # ── Section DiCE ────────────────────────────────────────────────────
        exp = load_dice_explainer(model, transformer)
        render_dice_section(exp=exp, input_df=input_df)

    # ── Chat widget ──────────────────────────────────────────────────────────
    render_chat_widget()


if __name__ == "__main__":
    main()