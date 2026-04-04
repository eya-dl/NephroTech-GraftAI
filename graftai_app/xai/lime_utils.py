# ═══════════════════════════════════════════════════════
#  xai/lime_utils.py — Calcul et rendu LIME
# ═══════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import streamlit as st
import lime
import lime.lime_tabular

from config import DEFAULT_LIME_SAMPLES, DEFAULT_LIME_FEATURES
from plots import plot_lime_explanation
from xai.shap_utils import get_feature_names_after_ohe


def compute_lime(
    model,
    X_train: np.ndarray,
    input_vec: np.ndarray,
    transformer=None,                   # ✅ NOUVEAU — pour récupérer les vrais noms
    feature_names: list[str] = None,    # fallback si pas de transformer
    num_samples: int = DEFAULT_LIME_SAMPLES,
    num_features: int = DEFAULT_LIME_FEATURES,
):
    """
    Construit un LimeTabularExplainer et explique l'instance input_vec.
    Utilise les noms de features après OHE si le transformer est fourni.

    Retourne l'objet LimeExplanation (avec .as_list(), .score, etc.)
    """
    # ✅ Récupérer les vrais noms de features alignés sur X_train (après OHE)
    if transformer is not None:
        transformed_names = get_feature_names_after_ohe(transformer)
    elif feature_names is not None:
        transformed_names = feature_names
    else:
        transformed_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=transformed_names,   # ✅ noms alignés
        class_names=["Pas de rejet", "Rejet"],
        mode="classification",
        discretize_continuous=True,
    )
    explanation = explainer.explain_instance(
        input_vec,
        model.predict_proba,
        num_features=num_features,
        num_samples=num_samples,
    )
    return explanation


def build_lime_dataframe(lime_exp) -> pd.DataFrame:
    """
    Construit un DataFrame lisible à partir d'un LimeExplanation,
    trié par |impact| décroissant.
    """
    df = pd.DataFrame(lime_exp.as_list(), columns=["Règle (condition)", "Impact"])
    df["Direction"] = df["Impact"].apply(
        lambda v: "↑ Aggravant" if v > 0 else "↓ Protecteur"
    )
    df["_abs"] = df["Impact"].abs()
    df = df.sort_values("_abs", ascending=False).drop(columns="_abs")
    df["Impact"] = df["Impact"].apply(lambda v: f"{v:+.4f}")
    return df.reset_index(drop=True)


def render_lime_tab(
    model,
    X_train: np.ndarray,
    input_vec: np.ndarray,
    feature_names: list[str],           # ALL_FEATURE_NAMES — gardé pour compatibilité
    transformer=None,                   # ✅ NOUVEAU paramètre — passer le transformer joblib
    num_samples: int = DEFAULT_LIME_SAMPLES,
    num_features: int = DEFAULT_LIME_FEATURES,
) -> tuple:
    """
    Affiche le contenu complet de l'onglet LIME dans Streamlit.

    Retourne (lime_exp, fig_lime) pour la génération du rapport.
    """
    with st.spinner("Calcul LIME…"):
        lime_exp = compute_lime(
            model, X_train, input_vec,
            transformer=transformer,
            num_samples=num_samples,
            num_features=num_features,
        )

    st.markdown(
        "> **LIME** approxime localement le modèle par une régression linéaire "
        "interprétable, en perturbant les données d'entrée autour de "
        "l'instance à expliquer."
    )

    fig_lime = plot_lime_explanation(lime_exp, feature_names)
    st.pyplot(fig_lime, use_container_width=True)

    st.markdown("#### 📋 Règles LIME")
    lime_df = build_lime_dataframe(lime_exp)
    st.dataframe(lime_df, use_container_width=True, hide_index=True)

    score = lime_exp.score
    quality = (
        "Excellente" if score > 0.8 else
        "Bonne"      if score > 0.6 else
        "Acceptable"
    )
    st.info(
        f"📐 **Score de fidélité LIME** (R²) : {score:.3f} — {quality} approximation locale"
    )

    return lime_exp, fig_lime