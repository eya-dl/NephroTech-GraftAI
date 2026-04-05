# ═══════════════════════════════════════════════════════
#  xai/shap_utils.py — Calcul et rendu SHAP
# ═══════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import streamlit as st
import shap

from plots import plot_shap_waterfall, plot_shap_force


def get_feature_names_after_ohe(transformer) -> list[str]:
    """
    Récupère les noms de features APRÈS transformation OHE + Scaler.
    Indispensable car OHE expand les colonnes catégorielles.

    Retourne une liste alignée sur les colonnes de X_train_t.
    """
    feature_names = []

    for name, estimator, cols in transformer.transformers_:
        if name == "ohe":
            # OneHotEncoder → on récupère les catégories générées
            ohe_names = estimator.get_feature_names_out(cols)
            feature_names.extend(ohe_names.tolist())
        elif name == "scaler":
            # StandardScaler → on garde les noms originaux
            feature_names.extend(cols)
        # "remainder" ignoré

    return feature_names


def compute_shap(
    model,
    input_vec: np.ndarray,
    transformer=None,
) -> tuple[np.ndarray, float, list[str]]:
    """
    Calcule les valeurs SHAP pour une instance avec TreeExplainer.

    Retourne
    --------
    shap_values    : np.ndarray  — contributions par feature (classe 1)
    base_value     : float       — valeur de base (expected value du modèle)
    feature_names  : list[str]   — noms alignés sur shap_values
    """
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(input_vec.reshape(1, -1))

    # XGBoost / GBM retourne un tableau 2D (1 sample × n_features)
    if isinstance(raw, list):
        sv = raw[1][0]          # classe 1
    else:
        sv = raw[0]             # régression / sortie unique

    base = explainer.expected_value
    if isinstance(base, np.ndarray):
        base = float(base[1]) if len(base) > 1 else float(base[0])
    else:
        base = float(base)

    # ✅ Noms de features alignés sur le vecteur transformé
    if transformer is not None:
        feature_names = get_feature_names_after_ohe(transformer)
    else:
        feature_names = [f"feature_{i}" for i in range(len(sv))]

    return sv, base, feature_names


def build_shap_dataframe(
    shap_values: np.ndarray,
    feature_names: list[str],
    input_values: list[float],
    top_n: int = 12,
) -> pd.DataFrame:
    """
    Construit un DataFrame lisible des valeurs SHAP
    triées par |contribution| décroissante.
    """
    # ✅ Sécurité : aligner les longueurs
    n = min(len(shap_values), len(feature_names), len(input_values))

    df = pd.DataFrame({
        "Variable":       feature_names[:n],
        "Valeur patient": [f"{v:.2f}" for v in input_values[:n]],
        "SHAP":           [f"{v:+.4f}" for v in shap_values[:n]],
        "Direction":      ["↑ Aggravant" if v > 0 else "↓ Protecteur"
                           for v in shap_values[:n]],
    })
    df["_abs"] = np.abs(shap_values[:n])
    df = df.sort_values("_abs", ascending=False).drop(columns="_abs").head(top_n)
    return df.reset_index(drop=True)


def render_shap_tab(
    model,
    input_vec: np.ndarray,
    feature_names: list[str],        # ALL_FEATURE_NAMES (avant OHE) — gardé pour compatibilité
    input_values_full: list[float],
    transformer=None,                # ✅ NOUVEAU paramètre — passer le transformer joblib
    show_force_plot: bool = True,
) -> tuple[np.ndarray, float, "plt.Figure"]:
    """
    Affiche le contenu complet de l'onglet SHAP dans Streamlit.

    Retourne (shap_values, base_value, fig_shap) pour la génération du rapport.
    """
    with st.spinner("Calcul SHAP…"):
        # ✅ On passe le transformer pour récupérer les vrais noms après OHE
        sv, base_val, transformed_names = compute_shap(model, input_vec, transformer)

    st.markdown(
        "> **SHAP** (SHapley Additive exPlanations) décompose la prédiction "
        "en contributions individuelles de chaque variable, "
        "basées sur la théorie des jeux coopératifs."
    )

    # ✅ input_values_full doit aussi être aligné sur le vecteur transformé
    input_vals_transformed = input_vec.flatten().tolist()

    fig_shap = plot_shap_waterfall(sv, transformed_names, input_vals_transformed, base_val)
    st.pyplot(fig_shap, use_container_width=True)

    if show_force_plot:
        st.markdown("#### Force Plot")
        prob = float(model.predict_proba(input_vec.reshape(1, -1))[0, 1])
        fig_force = plot_shap_force(sv, transformed_names, base_val, prob)
        st.pyplot(fig_force, use_container_width=True)

    st.markdown("#### Tableau des valeurs SHAP")
    shap_df = build_shap_dataframe(sv, transformed_names, input_vals_transformed)
    st.dataframe(shap_df, use_container_width=True, hide_index=True)

    return sv, base_val, fig_shap