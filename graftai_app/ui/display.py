# ═══════════════════════════════════════════════════════
#  ui/display.py — Affichage des résultats, onglets XAI,
#                  section rapport
# ═══════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from xai.shap_utils import get_feature_names_after_ohe

from config import ALL_FEATURE_NAMES
from plots import (
    plot_risk_gauge,
    plot_feature_importance_global,
    plot_probability_distribution,
)
from utils.helpers import (
    risk_label,
    risk_emoji,
    confidence_interval,
    predicted_class,
    patient_percentile,
)
from utils.report import generate_html_report
from xai.shap_utils import render_shap_tab
from xai.lime_utils import render_lime_tab


# ── En-tête principal ────────────────────────────────────

def render_header() -> None:
    """Affiche le bandeau titre de l'application."""
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1.5rem 0">
      <div style="font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;
                  color:#8895b3;margin-bottom:.5rem">
        SYSTÈME D'AIDE À LA DÉCISION MÉDICALE
      </div>
      <h1 style="font-size:2.2rem;font-weight:800;
                 background:linear-gradient(135deg,#00c6ff,#7b61ff);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0">
        🫀 GraftAI
      </h1>
      <p style="color:#8895b3;margin-top:.4rem;font-size:.95rem">
        Prédiction du Risque de Rejet de Greffon — Intelligence Artificielle Explicable
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()


# ── Résultat de prédiction ───────────────────────────────

def render_prediction_result(prob: float) -> None:
    """
    Affiche la jauge de risque et les métriques associées
    (IC, seuil, classe prédite).
    """
    st.markdown("##  Résultat de la Prédiction")

    r_label = risk_label(prob)
    r_emoji = risk_emoji(prob)
    ic_low, ic_high = confidence_interval(prob)

    col_gauge, col_meta = st.columns([1, 2])

    with col_gauge:
        fig_gauge = plot_risk_gauge(prob)
        st.pyplot(fig_gauge, use_container_width=True)

    with col_meta:
        st.metric(
            "Probabilité de rejet",
            f"{prob * 100:.1f}%",
            delta=f"Risque {r_label} {r_emoji}",
        )

        if prob < 0.33:
            st.success(f"**{r_emoji} Risque FAIBLE** — Profil favorable. Maintenir la surveillance standard.")
        elif prob < 0.66:
            st.warning(f"**{r_emoji} Risque MODÉRÉ** — Renforcer la surveillance et ajuster le traitement.")
        else:
            st.error(f"**{r_emoji} Risque ÉLEVÉ** — Intervention urgente recommandée. Consulter un spécialiste.")

        st.markdown(f"""
        <div style="background:rgba(25,34,64,.6);border:1px solid rgba(0,198,255,.15);
                    border-radius:10px;padding:1rem;margin-top:.5rem;
                    font-size:.82rem;color:#8895b3">
          <strong style="color:#e8eaf6">Interprétation :</strong><br>
          IC 95% estimé : [{ic_low * 100:.1f}% — {ic_high * 100:.1f}%]<br>
          Seuil de décision : 0.50 (standard) | 0.33 (conservateur)<br>
          Classe prédite :
          <span style="color:#00c6ff;font-weight:600">{predicted_class(prob)}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    return fig_gauge


# ── Onglets XAI ──────────────────────────────────────────

def render_xai_tabs(
    model,
    X_train: np.ndarray,
    input_vec: np.ndarray,
    input_values_full: list[float],
    n_lime_samples: int,
    n_lime_features: int,
    show_force_plot: bool,
    show_distribution: bool,
    transformer=None, 
) -> dict:
    """
    Affiche les 4 onglets XAI et retourne un dict contenant
    les objets nécessaires à la génération du rapport :

    {
        "shap_values" : np.ndarray,
        "base_value"  : float,
        "lime_exp"    : LimeExplanation,
        "fig_shap"    : plt.Figure,
        "fig_lime"    : plt.Figure,
        "fig_global"  : plt.Figure,
        "fig_dist"    : plt.Figure | None,
    }
    """
    st.markdown("## Résultats d'Explicabilité (XAI)")

    tab_shap, tab_lime, tab_global, tab_dist = st.tabs([
        "  SHAP Analysis",
        "  LIME Analysis",
        "  Importance Globale",
        "  Distribution",
    ])

    # ── SHAP ──────────────────────────────────────────────
    with tab_shap:
        sv, base_val, fig_shap = render_shap_tab(
            model, input_vec, ALL_FEATURE_NAMES, input_values_full,
            show_force_plot=show_force_plot,transformer=transformer,
        )

    # ── LIME ──────────────────────────────────────────────
    with tab_lime:
        lime_exp, fig_lime = render_lime_tab(
            model, X_train, input_vec, ALL_FEATURE_NAMES,
            num_samples=n_lime_samples, num_features=n_lime_features,transformer=transformer,
        )

    # ── Importance globale ────────────────────────────────
    with tab_global:

        

        if transformer is not None:
            transformed_names = get_feature_names_after_ohe(transformer)
        else:
            transformed_names = ALL_FEATURE_NAMES   # fallback

        fig_global = plot_feature_importance_global(model, transformed_names)
        st.pyplot(fig_global, use_container_width=True)
        

        importances = model.feature_importances_
        n = min(len(transformed_names), len(importances))
        imp_df = pd.DataFrame({
            "Variable":   transformed_names[:n],
            "Importance": importances[:n],
        }).sort_values("Importance", ascending=False).head(15)
        imp_df["Importance"] = imp_df["Importance"].apply(lambda v: f"{v:.4f}")
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

    # ── Distribution ──────────────────────────────────────
    fig_dist = None
    with tab_dist:
        if show_distribution:
            fig_dist = plot_probability_distribution(model, X_train, input_vec)
            st.pyplot(fig_dist, use_container_width=True)

            all_probs = model.predict_proba(X_train)[:, 1]
            prob      = float(model.predict_proba(input_vec.reshape(1, -1))[0, 1])
            pct       = patient_percentile(prob, all_probs)
            st.metric(
                "Percentile du patient", f"{pct:.0f}ème",
                help="Position du patient dans la distribution des risques de la population",
            )
        else:
            st.info("Activez 'Afficher distribution population' dans la barre latérale.")

    return {
        "shap_values": sv,
        "base_value":  base_val,
        "lime_exp":    lime_exp,
        "fig_shap":    fig_shap,
        "fig_lime":    fig_lime,
        "fig_global":  fig_global,
        "fig_dist":    fig_dist,
    }


# ── Section rapport ──────────────────────────────────────

def render_report_section(
    patient_info: dict,
    prob: float,
    xai_results: dict,
    input_values_full: list[float],
    fig_gauge,
    model,
    X_train,
    input_vec,
    transformer=None,
) -> None:
    """
    Affiche la section de génération du rapport HTML
    et propose le téléchargement.
    """
    st.divider()
    st.markdown("## 📄 Génération du Rapport d'Explicabilité")

    col_info, col_btn = st.columns([2, 1])
    with col_info:
        st.markdown("""
        Le rapport inclut :
        -  Score de risque avec jauge visuelle
        -  Analyse SHAP complète (waterfall + tableaux)
        -  Analyse LIME complète (règles + graphiques)
        -  Importance globale des variables
        -  Distribution population de référence
        -  Recommandations cliniques
        -  Avertissements réglementaires
        """)

    with col_btn:
        generate = st.button(" Générer le Rapport HTML", use_container_width=True)

    if generate:
        with st.spinner(" Génération du rapport XAI complet…"):
            # Si fig_dist n'a pas été créée (option désactivée), on la génère quand même
            fig_dist = xai_results["fig_dist"] or plot_probability_distribution(
                model, X_train, input_vec
            )
            if transformer is not None:
                feature_names = get_feature_names_after_ohe(transformer)
                input_values_aligned = transformer.transform(input_vec.reshape(1, -1)).flatten().tolist()
            else:
                feature_names = ALL_FEATURE_NAMES
                input_values_aligned = input_values_full
            
            input_transformed = transformer.transform(input_vec.reshape(1, -1))
            input_values_aligned = input_transformed.flatten().tolist()

            html = generate_html_report(
                patient_info=patient_info,
                prob=prob,
                shap_values=xai_results["shap_values"],
                lime_exp=xai_results["lime_exp"],
                feature_names=feature_names,
                input_values=input_values_aligned,
                fig_gauge=fig_gauge,
                fig_shap=xai_results["fig_shap"],
                fig_lime=xai_results["fig_lime"],
                fig_global=xai_results["fig_global"],
                fig_dist=fig_dist,
            )

        patient_id = patient_info.get("id", "patient").replace("-", "_")
        filename   = f"rapport_xai_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"

        st.download_button(
            label=" Télécharger le Rapport HTML",
            data=html.encode("utf-8"),
            file_name=filename,
            mime="text/html",
            use_container_width=True,
        )
        st.success(f" Rapport généré : `{filename}`")

        with st.expander(" Aperçu (extrait HTML)"):
            st.code(html[:2000] + "\n…[tronqué]", language="html")