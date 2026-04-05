# ═══════════════════════════════════════════════════════
#  config.py — Configuration GraftAI (features réelles)
# ═══════════════════════════════════════════════════════

APP_TITLE   = "GraftAI"
APP_ICON    = "🏥"
APP_VERSION = "1.0.0"
RANDOM_SEED = 42

# ── Features numériques (float64 / int64) ────────────────
NUMERICAL_FEATURES = {
 
    # ── Patient ─────────────────────────────────────────────
    "age": {
        "type": "int", "min": 0, "max": 100,
        "default": 45, "unit": "ans",
        "label": "Âge du patient"
    },
    "c_PRA": {
        "type": "float", "min": 0.0, "max": 100.0,
        "default": 0.0, "unit": "%",
        "label": "cPRA — Panel Reactive Antibody"
    },
    "dialysis_time": {
        "type": "float", "min": 0.0, "max": 300.0,
        "default": 24.0, "unit": "mois",
        "label": "Durée de dialyse avant greffe"
    },
    "n_pregnant": {
        "type": "int", "min": 0, "max": 20,
        "default": 0, "unit": "",
        "label": "Nombre de grossesses (0 si homme)"
    },
 
    # ── Donneur ──────────────────────────────────────────────
    "age_donor": {
        "type": "int", "min": 0, "max": 100,
        "default": 42, "unit": "ans",
        "label": "Âge du donneur"
    },
    "final_creatinine_donor": {
        "type": "float", "min": 0.0, "max": 20.0,
        "default": 1.6, "unit": "mg/dL",
        "label": "Créatinine finale du donneur"
    },
    "cold_ischemia_time": {
        "type": "int", "min": 0, "max": 30,
        "default": 1, "unit": "jours",
        "label": "Temps d'ischémie froide"
    },
 
    # ── Post-greffe ──────────────────────────────────────────
    "lenght_stay": {
        "type": "float", "min": 0.0, "max": 365.0,
        "default": 17.0, "unit": "jours",
        "label": "Durée d'hospitalisation"
    },
    "creatinine_discharge": {
        "type": "float", "min": 0.0, "max": 20.0,
        "default": 10.0, "unit": "mg/dL",
        "label": "Créatinine à la sortie (patient)"
    },
    "MM": {
        "type": "int", "min": 0, "max": 6,
        "default": 3, "unit": "",
        "label": "HLA Mismatch (0–6)"
    },
 
    # ── Suivi MAT / année ────────────────────────────────────
    "time_mat": {
        "type": "int", "min": 0, "max": 500,
        "default": 7, "unit": "jours",
        "label": "Délai d'apparition MAT post-greffe"
    },
    "year": {
        "type": "int", "min": 2000, "max": 2030,
        "default": 2014, "unit": "",
        "label": "Année de transplantation"
    },
    
}
 
# ══════════════════════════════════════════════════════════════
#  FEATURES CATÉGORIELLES  (object)
#  Format : dict de listes simples — compatible avec app.py
#  Ordre exact issu de df.dtypes  |  rejection exclu (cible)
# ══════════════════════════════════════════════════════════════
CATEGORICAL_FEATURES = {
 
    # ── Patient ──────────────────────────────────────────────
    "sex":                  ["M", "F"],
    "race":                 ["W", "B"],
    "underline_disease":    ["SHUa", "GN", "Nefropatia C3", "indeterminate"],
    "dialysis_type":        ["HD", "DP", "conservative"],
 
    # ── Donneur ──────────────────────────────────────────────
    "donor_type":           ["deceased", "alive"],
    "donor_sex":            ["M", "F"],
    "donor_death":          ["CerebroVascular", "TCE", "other"],
 
    # ── Complications immédiates ─────────────────────────────
    "DGF":                  ["yes", "no"],
 
    # ── Immunosuppression ────────────────────────────────────
    "induction":            ["thymoglobulin", "basiliximab", "other", "no"],
    "induction_eculizumab": ["no", "yes"],
    "immunosuppresion":     ["tac_mmf_pdn", "tac_mtor_dn", "other"],
    "convertion":           ["no", "yes"],
 
    # ── Traitements post-greffe ──────────────────────────────
    "plasmaferesis":        ["no", "yes"],
    "eculizumab":           ["no", "prophylactic", "treatment"],
    "enxertectomy":         ["no", "yes"],
 
    # ── Outcomes ─────────────────────────────────────────────
    "TRS":                  ["no", "yes"],
    "death":                ["no", "yes"],
    "cause.death":          [
        "still alive",
        "septic shock",
        "septic shock + hemorrhagic shock",
        "hemorrhagic/septic shock",
    ],
 
    # ── Génétique ────────────────────────────────────────────
    "gene":                 [
        "not identified", "NR", "no mutation",
        "C3", "CFH", "CFHR5", "CFI, CFB",
        "CFI, CFHR1, CFHR3", "CFH, CFI",
        "CFHR5, THBD", "CFHR5, CFHR1",
        "CFHR1 e CFHR3", "THBD, PLG", "other",
    ],
 
    
    # ── Traitement combiné ───────────────────────────────────
    "trat_combined":        [
        "no treatment",
        "only_plasmapheresis",
        "only eculizumab",
        "plasmapheresis and eculizumab",
    ],
}
 
# ── Labels lisibles pour l'UI ────────────────────────────
CATEGORICAL_LABELS = {
    "sex":                  "Sexe du patient",
    "race":                 "Race",
    "underline_disease":    "Maladie sous-jacente",
    "dialysis_type":        "Type de dialyse",
    "donor_type":           "Type de donneur",
    "donor_sex":            "Sexe du donneur",
    "donor_death":          "Cause du décès du donneur",
    "DGF":                  "DGF — Delayed Graft Function",
    "induction":            "Induction immunosuppressive",
    "induction_eculizumab": "Induction éculizumab",
    "immunosuppresion":     "Régime d'immunosuppression",
    "convertion":           "Conversion du traitement IS",
    "plasmaferesis":        "Plasmaphérèse",
    "eculizumab":           "Éculizumab",
    "enxertectomy":         "Enxertectomie (ablation du greffon)",
    "TRS":                  "Retour en Traitement de Suppléance",
    "death":                "Décès",
    "cause.death":          "Cause du décès",
    "gene":                 "Gène muté identifié",
    "trat_combined":        "Traitement combiné post-greffe",
}
 
# ── Features à fort impact sur la prédiction ─────────────
IMPORTANT_FEATURES = [
    "underline_disease",
    "induction",
    "c_PRA",
    "MM",
]
 
# ── Colonne cible ────────────────────────────────────────
TARGET = "rejection"

# ── Toutes les features dans l'ordre ────────────────────
ALL_FEATURE_NAMES = list(NUMERICAL_FEATURES.keys()) + list(CATEGORICAL_FEATURES.keys())

# ── Seuils de risque ─────────────────────────────────────
DECISION_THRESHOLD = 0.5
RISK_LOW           = 0.3
RISK_MODERATE      = 0.6

# ── Paramètres LIME ──────────────────────────────────────
DEFAULT_LIME_SAMPLES  = 500
DEFAULT_LIME_FEATURES = 10

# ── Affichage SHAP ───────────────────────────────────────
MAX_SHAP_DISPLAY = 15

# ── Recommandations cliniques ────────────────────────────
CLINICAL_RECOMMENDATIONS = {
    "low": """
        <p><strong style="color:#00e5b0">Risque faible de rejet.</strong></p>
        <p>Le modèle prédit un risque faible de rejet du greffon. Maintenir le protocole
        immunosuppresseur standard et assurer un suivi régulier.</p>
        <p>Recommandations : consultation de contrôle dans 3 mois, bilan biologique standard.</p>
    """,
    "moderate": """
        <p><strong style="color:#ffb347">⚠️ Risque modéré de rejet.</strong></p>
        <p>Le modèle identifie un risque modéré. Une surveillance rapprochée est conseillée
        ainsi qu'une réévaluation du protocole immunosuppresseur.</p>
        <p>Recommandations : consultation dans 4–6 semaines, dosage des inhibiteurs de calcineurine,
        biopsie à considérer selon l'évolution clinique.</p>
    """,
    "high": """
        <p><strong style="color:#ff4d6d">🔴 Risque élevé de rejet.</strong></p>
        <p>Le modèle prédit un risque élevé de rejet du greffon. Une prise en charge urgente
        est recommandée avec réévaluation immédiate du traitement.</p>
        <p>Recommandations : hospitalisation à envisager, biopsie rénale en urgence,
        ajustement immunosuppresseur immédiat, avis néphrologique spécialisé.</p>
    """,
}

# ── Couleurs thème sombre ────────────────────────────────
COLORS = {
    "bg":      "#0e1117",
    "bg2":     "#1a1d27",
    "surface": "#22263a",
    "accent":  "#7c9cff",
    "text":    "#e8eaf6",
    "text2":   "#9098b8",
    "safe":    "#00e5b0",
    "warn":    "#ffb347",
    "danger":  "#ff4d6d",
}

def rgba(hex_color: str, alpha: float = 1.0) -> tuple:
    """Convertit un hex (#rrggbb) en tuple RGBA pour matplotlib."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
    return (r, g, b, alpha)