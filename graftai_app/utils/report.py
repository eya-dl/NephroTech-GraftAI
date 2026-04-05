# ═══════════════════════════════════════════════════════
#  utils/report.py — Génération du rapport HTML d'explicabilité
# ═══════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from config import (
    APP_VERSION,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    CLINICAL_RECOMMENDATIONS,
)
from utils.helpers import (
    fig_to_base64,
    risk_level,
    risk_label,
    risk_color_hex,
    confidence_interval,
)


# ── Constructeurs de lignes HTML ─────────────────────────

def _attr_rows(feature_names: list[str], input_values: list[float]) -> str:
    """Construit les lignes <tr> du tableau des attributs cliniques."""
    rows = ""
    num_keys  = list(NUMERICAL_FEATURES.keys())
    cat_items = list(CATEGORICAL_FEATURES.items())
    n_num     = len(num_keys)

    for i, (feat, val) in enumerate(zip(feature_names[:n_num], input_values[:n_num])):
        unit = list(NUMERICAL_FEATURES.values())[i]["unit"]
        rows += f"<tr><td>{feat}</td><td style='text-align:center;font-weight:500'>{val:.2f} {unit}</td></tr>"

    for j, (feat, opts) in enumerate(cat_items):
        idx  = int(input_values[n_num + j])
        val  = opts[min(idx, len(opts) - 1)]
        rows += f"<tr><td>{feat}</td><td style='text-align:center;font-weight:500'>{val}</td></tr>"

    return rows


def _shap_rows(
    shap_values: np.ndarray,
    feature_names: list[str],
    input_values: list[float],
    top_n: int = 8,
) -> str:
    """Construit les lignes <tr> du tableau SHAP."""
    order = np.argsort(np.abs(shap_values))[::-1][:top_n]
    rows  = ""
    for i in order:
        name  = feature_names[i]
        val   = shap_values[i]
        inp   = input_values[i]
        color = "#ff4d6d" if val > 0 else "#00e5b0"
        dir_  = "↑ Aggravant" if val > 0 else "↓ Protecteur"
        rows += f"""
        <tr>
          <td>{name}</td>
          <td style="text-align:center">{inp:.2f}</td>
          <td style="text-align:center;color:{color};font-weight:600">{val:+.4f}</td>
          <td style="text-align:center;color:{color}">{dir_}</td>
        </tr>"""
    return rows


def _lime_rows(lime_exp, top_n: int = 8) -> str:
    """Construit les lignes <tr> du tableau LIME."""
    rows = ""
    for label, val in sorted(lime_exp.as_list(), key=lambda x: abs(x[1]), reverse=True)[:top_n]:
        color = "#ff4d6d" if val > 0 else "#00e5b0"
        rows += f"""
        <tr>
          <td>{label}</td>
          <td style="text-align:center;color:{color};font-weight:600">{val:+.4f}</td>
          <td style="text-align:center;color:{color}">{'Aggravant' if val > 0 else 'Protecteur'}</td>
        </tr>"""
    return rows


# ── CSS embarqué ─────────────────────────────────────────

def _report_css(risk_color: str) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',sans-serif;background:#0a0f1e;color:#e8eaf6;padding:2rem;line-height:1.6}}
.header{{background:linear-gradient(135deg,#0f1729,#192240);border:1px solid rgba(0,198,255,0.2);
         border-radius:16px;padding:2rem 2.5rem;margin-bottom:2rem;
         display:flex;justify-content:space-between;align-items:center}}
.header h1{{font-size:1.8rem;font-weight:700;color:#e8eaf6}}
.header h1 span{{color:#00c6ff}}
.meta{{color:#8895b3;font-size:0.85rem;text-align:right}}
.chip{{display:inline-block;background:rgba(0,198,255,0.1);border:1px solid rgba(0,198,255,0.25);
       border-radius:6px;padding:.15rem .5rem;font-size:.75rem;color:#00c6ff;font-family:monospace;margin:2px}}
.risk-banner{{background:linear-gradient(135deg,rgba(25,34,64,.9),rgba(20,30,53,.9));
             border:2px solid {risk_color}40;border-radius:16px;padding:2rem;
             margin-bottom:2rem;text-align:center}}
.risk-score{{font-size:4rem;font-weight:800;color:{risk_color}}}
.risk-label{{font-size:1.2rem;color:{risk_color};font-weight:600;letter-spacing:.15em;
            text-transform:uppercase;margin-top:.5rem}}
.section{{background:rgba(25,34,64,.6);border:1px solid rgba(0,198,255,.12);
          border-radius:12px;padding:1.5rem;margin-bottom:1.5rem}}
.section h2{{color:#00c6ff;font-size:.75rem;text-transform:uppercase;letter-spacing:.15em;
            font-weight:600;margin-bottom:1rem;padding-bottom:.5rem;
            border-bottom:1px solid rgba(0,198,255,.15)}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{background:rgba(0,198,255,.1);color:#8895b3;font-size:.7rem;text-transform:uppercase;
    letter-spacing:.1em;padding:.6rem .8rem;text-align:left;font-weight:600}}
td{{padding:.55rem .8rem;border-bottom:1px solid rgba(0,198,255,.07);color:#e8eaf6}}
tr:hover td{{background:rgba(0,198,255,.04)}}
.img-container{{text-align:center;margin:1rem 0}}
img{{max-width:100%;border-radius:10px;border:1px solid rgba(0,198,255,.1)}}
.recommendation{{background:rgba(25,34,64,.8);border-left:4px solid {risk_color};
                border-radius:0 12px 12px 0;padding:1.2rem 1.5rem}}
.recommendation p{{margin-bottom:.5rem;font-size:.9rem;color:#c8d0e0}}
.warn-box{{background:rgba(255,77,109,.1);border:1px solid rgba(255,77,109,.3);
           border-radius:8px;padding:1rem;margin-top:1rem;font-size:.8rem;color:#ffb3be}}
.footer{{text-align:center;color:#4a5568;font-size:.75rem;margin-top:2rem;
         padding-top:1rem;border-top:1px solid rgba(0,198,255,.1)}}
</style>"""


# ── Fonction principale ──────────────────────────────────

def generate_html_report(
    patient_info: dict,
    prob: float,
    shap_values: np.ndarray,
    lime_exp,
    feature_names: list[str],
    input_values: list[float],
    fig_gauge: plt.Figure,
    fig_shap: plt.Figure,
    fig_lime: plt.Figure,
    fig_global: plt.Figure,
    fig_dist: plt.Figure,
) -> str:
    """
    Génère un rapport HTML complet auto-contenu (images embarquées en base64).

    Paramètres
    ----------
    patient_info : dict avec clés 'nom', 'id', 'medecin'
    prob         : probabilité de rejet (float 0–1)
    shap_values  : array SHAP (1D, n_features)
    lime_exp     : objet LimeExplanation
    feature_names: liste des noms de features
    input_values : valeurs du vecteur patient (float list)
    fig_*        : figures matplotlib
    """
    now        = datetime.now().strftime("%d/%m/%Y à %H:%M")
    r_level    = risk_level(prob)
    r_label    = risk_label(prob)
    r_color    = risk_color_hex(prob)
    ic_low, ic_high = confidence_interval(prob)
    reco       = CLINICAL_RECOMMENDATIONS[r_level]

    attr_rows  = _attr_rows(feature_names, input_values)
    shap_rows_ = _shap_rows(shap_values, feature_names, input_values)
    lime_rows_ = _lime_rows(lime_exp)

    # Encode all figures
    b64 = {k: fig_to_base64(v) for k, v in {
        "gauge":  fig_gauge,
        "shap":   fig_shap,
        "lime":   fig_lime,
        "global": fig_global,
        "dist":   fig_dist,
    }.items()}

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Rapport XAI — Risque de Rejet de Greffon</title>
{_report_css(r_color)}
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div>
    <div style="color:#8895b3;font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;margin-bottom:.4rem">
      RAPPORT D'EXPLICABILITÉ IA — GraftAI {APP_VERSION}
    </div>
    <h1>Graft<span>AI</span> — Risque de Rejet de Greffon</h1>
    <div style="margin-top:.5rem">
      <span class="chip">GradientBoosting</span>
      <span class="chip">SHAP</span>
      <span class="chip">LIME</span>
      <span class="chip">XAI Médical</span>
    </div>
  </div>
  <div class="meta">
    <div style="font-weight:600;color:#e8eaf6;font-size:.95rem">{patient_info.get('nom','—')}</div>
    <div>ID&nbsp;: {patient_info.get('id','—')}</div>
    <div>Médecin&nbsp;: {patient_info.get('medecin','—')}</div>
    <div style="margin-top:.5rem">{now}</div>
  </div>
</div>

<!-- SCORE DE RISQUE -->
<div class="risk-banner">
  <div style="color:#8895b3;font-size:.75rem;letter-spacing:.15em;text-transform:uppercase;margin-bottom:.5rem">
    Score de risque calculé par l'IA
  </div>
  <div class="risk-score">{prob * 100:.1f}%</div>
  <div class="risk-label">Risque {r_label}</div>
  <div style="color:#8895b3;font-size:.8rem;margin-top:.5rem">
    IC 95% estimé&nbsp;: [{ic_low * 100:.1f}% — {ic_high * 100:.1f}%]
  </div>
  <div class="img-container" style="margin-top:1rem">
    <img src="data:image/png;base64,{b64['gauge']}" style="max-width:400px"/>
  </div>
</div>

<!-- ATTRIBUTS -->
<div class="section">
  <h2> Attributs cliniques du patient</h2>
  <table>
    <thead><tr><th>Variable</th><th>Valeur</th></tr></thead>
    <tbody>{attr_rows}</tbody>
  </table>
</div>

<!-- RECOMMANDATION -->
<div class="section">
  <h2> Recommandation clinique</h2>
  <div class="recommendation">{reco}</div>
</div>

<!-- SHAP -->
<div class="section">
  <h2> Analyse SHAP (SHapley Additive exPlanations)</h2>
  <p style="color:#8895b3;font-size:.85rem;margin-bottom:1rem">
    Les valeurs SHAP quantifient la contribution de chaque variable à la prédiction
    via la théorie des jeux coopératifs. Rouge = aggravant, vert = protecteur.
  </p>
  <div class="img-container">
    <img src="data:image/png;base64,{b64['shap']}"/>
  </div>
  <h2 style="margin-top:1.5rem">Top facteurs SHAP</h2>
  <table>
    <thead>
      <tr>
        <th>Variable</th>
        <th style="text-align:center">Valeur patient</th>
        <th style="text-align:center">SHAP</th>
        <th style="text-align:center">Direction</th>
      </tr>
    </thead>
    <tbody>{shap_rows_}</tbody>
  </table>
</div>

<!-- LIME -->
<div class="section">
  <h2> Analyse LIME (Local Interpretable Model-agnostic Explanations)</h2>
  <p style="color:#8895b3;font-size:.85rem;margin-bottom:1rem">
    LIME génère un modèle linéaire local par perturbation autour de l'instance.
    Les règles indiquent les intervalles de valeurs influençant la décision.
  </p>
  <div class="img-container">
    <img src="data:image/png;base64,{b64['lime']}"/>
  </div>
  <h2 style="margin-top:1.5rem"> Règles LIME</h2>
  <table>
    <thead>
      <tr>
        <th>Règle (condition)</th>
        <th style="text-align:center">Impact</th>
        <th style="text-align:center">Direction</th>
      </tr>
    </thead>
    <tbody>{lime_rows_}</tbody>
  </table>
</div>

<!-- IMPORTANCE GLOBALE -->
<div class="section">
  <h2> Importance globale des variables</h2>
  <p style="color:#8895b3;font-size:.85rem;margin-bottom:1rem">
    Importance moyenne sur la population de référence (n=2 000 cas).
  </p>
  <div class="img-container">
    <img src="data:image/png;base64,{b64['global']}"/>
  </div>
</div>

<!-- DISTRIBUTION -->
<div class="section">
  <h2>📈 Position du patient dans la population</h2>
  <div class="img-container">
    <img src="data:image/png;base64,{b64['dist']}"/>
  </div>
</div>

<!-- MÉTHODOLOGIE -->
<div class="section">
  <h2>⚙️ Méthodologie & Avertissements</h2>
  <p style="color:#8895b3;font-size:.85rem;margin-bottom:.8rem">
    <strong style="color:#e8eaf6">Modèle :</strong>
    Gradient Boosting Classifier (n_estimators=200, max_depth=4) entraîné sur données synthétiques.
  </p>
  <p style="color:#8895b3;font-size:.85rem;margin-bottom:.8rem">
    <strong style="color:#e8eaf6">SHAP :</strong>
    TreeExplainer — propriétés d'efficience, symétrie, monotonie et dummy respectées.
  </p>
  <p style="color:#8895b3;font-size:.85rem">
    <strong style="color:#e8eaf6">LIME :</strong>
    Perturbation locale, noyau gaussien, 500 échantillons, 12 features actives.
  </p>
  <div class="warn-box">
     <strong>Avertissement :</strong> Ce rapport est généré par un SADM et ne remplace pas
    le jugement clinique. Les résultats doivent être interprétés dans le contexte global du patient.
    Ce système n'est pas un dispositif médical certifié CE.
  </div>
</div>

<div class="footer">
  GraftAI v{APP_VERSION} — Rapport généré le {now} — Confidentiel — Usage médical restreint
</div>

</body>
</html>"""

    return html