# ═══════════════════════════════════════════════════════
#  plots.py — Fonctions de visualisation matplotlib
#  Chaque fonction retourne une Figure prête pour st.pyplot()
# ═══════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingClassifier

from config import COLORS, rgba, MAX_SHAP_DISPLAY


# ── Helpers ──────────────────────────────────────────────

def _dark_axes(ax: plt.Axes) -> None:
    """Applique le thème sombre à un axe matplotlib."""
    ax.set_facecolor(COLORS["bg2"])
    for spine in ax.spines.values():
        spine.set_color(rgba(COLORS["accent"], 0.18))
    ax.tick_params(colors=COLORS["text2"], labelsize=9)
    ax.xaxis.label.set_color(COLORS["text2"])
    ax.yaxis.label.set_color(COLORS["text2"])


def _dark_fig(figsize: tuple = (10, 5)) -> tuple[plt.Figure, plt.Axes]:
    """Crée une Figure + Axes avec fond sombre."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS["bg2"])
    _dark_axes(ax)
    return fig, ax


def _legend(ax: plt.Axes, pos_label: str = "↑ Augmente le risque",
            neg_label: str = "↓ Réduit le risque") -> None:
    """Ajoute une légende danger/safe standardisée."""
    elems = [
        mpatches.Patch(facecolor=COLORS["danger"], alpha=0.85, label=pos_label),
        mpatches.Patch(facecolor=COLORS["safe"],   alpha=0.85, label=neg_label),
    ]
    ax.legend(
        handles=elems, loc="lower right", framealpha=0.2,
        facecolor=COLORS["surface"], edgecolor=rgba(COLORS["accent"], 0.3),
        labelcolor=COLORS["text"], fontsize=8.5,
    )


# ── Jauge de risque ──────────────────────────────────────

def plot_risk_gauge(prob: float) -> plt.Figure:
    """
    Jauge semi-circulaire affichant la probabilité de rejet.
    Zones : vert (< 33 %) → orange (33–66 %) → rouge (> 66 %).
    """
    fig, ax = plt.subplots(
        figsize=(5, 3),
        subplot_kw=dict(aspect="equal"),
        facecolor=COLORS["bg2"],
    )
    ax.set_facecolor(COLORS["bg2"])

    # Arc de fond
    theta = np.linspace(np.pi, 0, 200)
    for i in range(len(theta) - 1):
        t = i / (len(theta) - 1)
        color = (
            rgba(COLORS["safe"],   0.9) if t < 0.33 else
            rgba(COLORS["warn"],   0.9) if t < 0.66 else
            rgba(COLORS["danger"], 0.9)
        )
        ax.plot(
            [np.cos(theta[i]), np.cos(theta[i + 1])],
            [np.sin(theta[i]), np.sin(theta[i + 1])],
            color=color, linewidth=14, solid_capstyle="butt",
        )

    # Aiguille
    angle = np.pi - prob * np.pi
    ax.annotate(
        "", xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="white", lw=2.5, mutation_scale=15),
    )
    ax.plot(0, 0, "o", color="white", ms=8, zorder=10)

    # Labels zones
    for x, label, color in [
        (-0.95, "FAIBLE",  COLORS["safe"]),
        (0.00,  "MODÉRÉ",  COLORS["warn"]),
        (0.95,  "ÉLEVÉ",   COLORS["danger"]),
    ]:
        ax.text(x, -0.15, label, fontsize=8, color=color, ha="center",
                fontfamily="monospace", fontweight="bold")

    pct_color = (
        COLORS["safe"]   if prob < 0.33 else
        COLORS["warn"]   if prob < 0.66 else
        COLORS["danger"]
    )
    ax.text(0, 0.35, f"{prob * 100:.1f}%", fontsize=28, color=pct_color,
            ha="center", va="center", fontweight="bold")
    ax.text(0, 0.08, "RISQUE DE REJET", fontsize=8, color=COLORS["text2"],
            ha="center", va="center", fontfamily="monospace")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.4, 1.2)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ── SHAP Waterfall ───────────────────────────────────────

def plot_shap_waterfall(
    shap_values: np.ndarray,
    feature_names: list[str],
    input_values: list[float],
    expected_value: float,
) -> plt.Figure:
    """
    Bar chart horizontal des valeurs SHAP triées par |valeur|.
    Rouge = aggravant, vert = protecteur.
    """
    order = np.argsort(np.abs(shap_values))[::-1][:MAX_SHAP_DISPLAY]
    names  = [feature_names[i] for i in order][::-1]
    vals   = shap_values[order][::-1]

    fig, ax = _dark_fig(figsize=(10, 6))
    colors  = [
        rgba(COLORS["danger"], 0.85) if v > 0 else rgba(COLORS["safe"], 0.85)
        for v in vals
    ]
    ax.barh(range(len(names)), vals, color=colors, height=0.6, edgecolor="none")

    for i, v in enumerate(vals):
        x_off = 0.005 if v >= 0 else -0.005
        ax.text(v + x_off, i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8.5, color=COLORS["text"],
                fontfamily="monospace", fontweight="500")

    short = [n if len(n) <= 34 else n[:32] + "…" for n in names]
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(short, fontsize=9, color=COLORS["text"])
    ax.axvline(0, color=rgba(COLORS["accent"], 0.5), linewidth=1)
    ax.set_xlabel("Valeur SHAP (impact sur la prédiction)")
    ax.set_title("Contribution SHAP — Facteurs de risque",
                 color=COLORS["text"], fontsize=11, fontweight="600", pad=12)
    _legend(ax)
    plt.tight_layout()
    return fig


# ── SHAP Force Plot ──────────────────────────────────────

def plot_shap_force(
    shap_values: np.ndarray,
    feature_names: list[str],
    base_value: float,
    pred_value: float,
) -> plt.Figure:
    """
    Force plot simplifié : blocs rouge/vert empilés le long
    d'un axe base_value → pred_value.
    """
    pos = sorted(
        [(feature_names[i], shap_values[i]) for i in range(len(shap_values))
         if shap_values[i] > 0], key=lambda x: -x[1]
    )
    neg = sorted(
        [(feature_names[i], shap_values[i]) for i in range(len(shap_values))
         if shap_values[i] < 0], key=lambda x: x[1]
    )

    fig, ax = plt.subplots(figsize=(12, 2.8), facecolor=COLORS["bg2"])
    ax.set_facecolor(COLORS["bg2"])
    ax.axis("off")

    # Flèche base → prédiction
    ax.annotate(
        "", xy=(pred_value, 0.5), xytext=(base_value, 0.5),
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=2),
    )
    for x, label, color, fw in [
        (base_value, f"Base\n{base_value:.3f}", COLORS["text2"], "400"),
        (pred_value, f"Score\n{pred_value:.3f}", COLORS["text"],  "700"),
    ]:
        ax.text(x, 0.15, label, ha="center", color=color,
                fontsize=8.5, fontfamily="monospace", fontweight=fw)

    x = base_value
    for name, v in pos[:5]:
        short = name[:18] + "…" if len(name) > 18 else name
        ax.barh(0.5, v, left=x, height=0.3,
                color=COLORS["danger"], alpha=0.85, edgecolor="none")
        if v > 0.015:
            ax.text(x + v / 2, 0.5, short, ha="center", va="center",
                    fontsize=7.5, color="white", fontfamily="monospace")
        x += v

    x = base_value
    for name, v in neg[:5]:
        short = name[:18] + "…" if len(name) > 18 else name
        ax.barh(0.5, v, left=x + v, height=0.3,
                color=COLORS["safe"], alpha=0.85, edgecolor="none")
        if abs(v) > 0.015:
            ax.text(x + v / 2, 0.5, short, ha="center", va="center",
                    fontsize=7.5, color="white", fontfamily="monospace")
        x += v

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("SHAP Force Plot — Contributions cumulées",
                 color=COLORS["text"], fontsize=10, fontweight="600")
    plt.tight_layout()
    return fig


# ── LIME Bar Chart ───────────────────────────────────────

def plot_lime_explanation(lime_exp, feature_names: list[str]) -> plt.Figure:
    """
    Bar chart horizontal des importances locales LIME.
    exp_list est une liste [(règle_str, valeur), …].
    """
    exp_list = sorted(lime_exp.as_list(), key=lambda x: abs(x[1]))
    labels   = [e[0][:47] + "…" if len(e[0]) > 47 else e[0] for e in exp_list]
    values   = [e[1] for e in exp_list]
    colors   = [
        rgba(COLORS["danger"], 0.85) if v > 0 else rgba(COLORS["safe"], 0.85)
        for v in values
    ]

    fig, ax = _dark_fig(figsize=(10, max(4, len(labels) * 0.45)))
    ax.barh(range(len(labels)), values, color=colors, height=0.6, edgecolor="none")

    for i, v in enumerate(values):
        ax.text(
            v + (0.002 if v >= 0 else -0.002), i, f"{v:+.4f}",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=8.5, color=COLORS["text"], fontfamily="monospace",
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5, color=COLORS["text"])
    ax.axvline(0, color=rgba(COLORS["accent"], 0.5), linewidth=1)
    ax.set_xlabel("Importance locale LIME")
    ax.set_title("LIME — Explication locale de la prédiction",
                 color=COLORS["text"], fontsize=11, fontweight="600", pad=12)
    _legend(ax, "↑ Facteur aggravant", "↓ Facteur protecteur")
    plt.tight_layout()
    return fig


# ── Importance Globale ───────────────────────────────────

def plot_feature_importance_global(
    model: GradientBoostingClassifier, feature_names: list[str]
) -> plt.Figure:
    """
    Bar chart horizontal de l'importance moyenne des variables
    calculée par le modèle GBM (Gini impurity).
    """
    importances = model.feature_importances_
    feature_names = feature_names[:len(importances)]

    order = np.argsort(importances)[-15:]
    grad  = np.linspace(0, 1, len(order))
    colors_list = [
        (0, (198 * t + 123 * (1 - t)) / 255, (255 * t + 97 * (1 - t)) / 255, 0.85)
        for t in grad
    ]

    fig, ax = _dark_fig(figsize=(10, 6))
    ax.barh(range(len(order)), importances[order],
            color=colors_list, height=0.6, edgecolor="none")

    short = [
        feature_names[i][:35] + "…" if len(feature_names[i]) > 35
        else feature_names[i]
        for i in order
    ]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(short, fontsize=9, color=COLORS["text"])
    ax.set_xlabel("Importance (Gradient Boosting — Gini)")
    ax.set_title("Importance globale des variables — Modèle GBM",
                 color=COLORS["text"], fontsize=11, fontweight="600", pad=12)
    plt.tight_layout()
    return fig


# ── Distribution population ──────────────────────────────

def plot_probability_distribution(
    model: GradientBoostingClassifier,
    X_train: np.ndarray,
    input_vec: np.ndarray,
) -> plt.Figure:
    """
    Histogramme des probabilités de rejet sur la population de référence,
    avec la position du patient marquée par une ligne pointillée.
    """
    probs        = model.predict_proba(X_train)[:, 1]
    patient_prob = float(model.predict_proba(input_vec.reshape(1, -1))[0, 1])

    fig, ax = _dark_fig(figsize=(10, 4))
    counts, bins, patches = ax.hist(probs, bins=40, density=True, edgecolor="none")

    # Colorisation par zone de risque
    for patch, left, right in zip(patches, bins[:-1], bins[1:]):
        if right <= 0.33:
            patch.set_facecolor(rgba(COLORS["safe"],   0.45))
        elif right <= 0.66:
            patch.set_facecolor(rgba(COLORS["warn"],   0.45))
        else:
            patch.set_facecolor(rgba(COLORS["danger"], 0.45))

    # Ligne patient
    ax.axvline(patient_prob, color="white", linewidth=2.5, linestyle="--", zorder=5)
    ax.text(
        patient_prob + 0.01, ax.get_ylim()[1] * 0.85,
        f"Patient\n{patient_prob:.1%}", color="white", fontsize=9,
        fontfamily="monospace", fontweight="bold",
    )

    ax.set_xlabel("Probabilité de rejet")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution des risques — Population de référence (n=2 000)",
                 color=COLORS["text"], fontsize=11, fontweight="600", pad=12)
    plt.tight_layout()
    return fig