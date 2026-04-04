# ═══════════════════════════════════════════════════════
#  utils/helpers.py — Fonctions utilitaires transverses
# ═══════════════════════════════════════════════════════

import io
import base64
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from config import RISK_LOW, RISK_MODERATE, DECISION_THRESHOLD

# ── Injection CSS ────────────────────────────────────────

def inject_css() -> None:
    """
    Lit assets/styles.css et l'injecte dans la page Streamlit via st.markdown.
    Doit être appelée en début de main(), après set_page_config().
    """
    css_path = pathlib.Path(__file__).parent.parent / "assets" / "styles.css"
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ── Conversion figure → base64 ───────────────────────────

def fig_to_base64(fig: plt.Figure, dpi: int = 120) -> str:
    """
    Encode une Figure matplotlib en chaîne base64 (PNG).
    Utilisé pour intégrer les graphiques directement dans le rapport HTML.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Helpers de risque ────────────────────────────────────

def risk_level(prob: float) -> str:
    """Retourne 'low' | 'moderate' | 'high' selon la probabilité."""
    if prob < RISK_LOW:
        return "low"
    if prob < RISK_MODERATE:
        return "moderate"
    return "high"


def risk_label(prob: float) -> str:
    """Retourne le label français du niveau de risque."""
    return {"low": "FAIBLE", "moderate": "MODÉRÉ", "high": "ÉLEVÉ"}[risk_level(prob)]


def risk_emoji(prob: float) -> str:
    """Retourne l'emoji associé au niveau de risque."""
    return {"low": "✅", "moderate": "⚠️", "high": "🔴"}[risk_level(prob)]


def risk_color_hex(prob: float) -> str:
    """Retourne le code hex de la couleur associée au niveau de risque."""
    return {"low": "#00e5b0", "moderate": "#ffb347", "high": "#ff4d6d"}[risk_level(prob)]


def predicted_class(prob: float) -> str:
    """Retourne 'REJET' ou 'PAS DE REJET' selon le seuil de décision."""
    return "REJET" if prob >= DECISION_THRESHOLD else "PAS DE REJET"


def confidence_interval(prob: float, margin: float = 0.07) -> tuple[float, float]:
    """Intervalle de confiance estimé (±margin) clampé dans [0, 1]."""
    return max(0.0, prob - margin), min(1.0, prob + margin)


# ── Statistique population ───────────────────────────────

def patient_percentile(prob: float, all_probs: np.ndarray) -> float:
    """
    Retourne le percentile (0–100) du patient dans la population de référence.
    """
    return float((all_probs < prob).mean() * 100)