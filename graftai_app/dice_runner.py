# ═══════════════════════════════════════════════════════
#  dice_runner.py — Logique DiCE encapsulée pour GraftAI
# ═══════════════════════════════════════════════════════
import warnings
import joblib
import pandas as pd
import dice_ml

warnings.filterwarnings("ignore")

# ── Colonnes continues utilisées dans le dataset ────────────────────────────
CONTINUOUS_FEATURES = [
    'age', 'c_PRA', 'dialysis_time', 'n_pregnant', 'age_donor',
    'final_creatinine_donor', 'cold_ischemia_time', 'lenght_stay',
    'creatinine_discharge', 'MM', 'time_mat', 'year'
]

# ── Groupes de features modifiables ─────────────────────────────────────────
FEATURES_GROUP1 = [
    'immunosuppresion', 'convertion', 'plasmaferesis',
    'eculizumab', 'trat_combined'
]

FEATURES_GROUP2 = [
    'dialysis_type', 'dialysis_time', 'donor_type', 'age_donor',
    'donor_death', 'final_creatinine_donor', 'cold_ischemia_time',
    'lenght_stay', 'induction', 'induction_eculizumab',
    'eculizumab', 'enxertectomy'
]

FEATURES_GROUP3 = FEATURES_GROUP1 + FEATURES_GROUP2

# ── Mapping valeurs formulaire → valeurs dataset (uniquement pour DiCE) ──────
VALUE_MAPPING = {
    'race':      {'W': 'P'},
    'induction': {'thymoglobulin': 'thymogllubulin'},   # typo dans le dataset
}


def _translate_for_dice(patient_row: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige les valeurs du formulaire incompatibles avec le dataset.
    Travaille sur une copie — input_df original jamais modifié.
    """
    row = patient_row.copy()
    for col, mapping in VALUE_MAPPING.items():
        if col in row.columns:
            val = str(row[col].iloc[0])
            if val in mapping:
                row[col] = mapping[val]
    return row


def _validate_categorical_values(patient_row: pd.DataFrame, exp) -> None:
    """
    Affiche dans le terminal les valeurs catégorielles valides et invalides
    par rapport au dataset d'entraînement. Uniquement informatif.
    """
    print("\n===== VALIDATION VALEURS CATÉGORIELLES =====")
    for col in patient_row.columns:
        if col not in CONTINUOUS_FEATURES:
            val = str(patient_row[col].iloc[0])
            train_vals = sorted(exp.data_interface.data_df[col].astype(str).unique())
            if val not in train_vals:
                print(f"  ❌ {col}: '{val}' | valeurs acceptées : {train_vals}")
            else:
                print(f"  ✅ {col}: '{val}'")
    print("=============================================\n")


def load_train_raw(path: str = "train_dataset.xlsx", target_col: str = "rejection") -> pd.DataFrame:
    """
    Charge le dataset brut (pre-transform) depuis un fichier Excel.
    """
    df = pd.read_excel(path)

    # ── Nettoyage noms colonnes ──────────────────────────────────────────────
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('\xa0', '', regex=False)
    df.columns = df.columns.str.replace('\u200b', '', regex=False)

    # ── Vérification colonne cible ───────────────────────────────────────────
    if target_col not in df.columns:
        raise ValueError(
            f"Colonne cible '{target_col}' introuvable dans '{path}'.\n"
            f"Colonnes disponibles : {list(df.columns)}"
        )

    # ── Forcer les colonnes continues en numérique ───────────────────────────
    for col in CONTINUOUS_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Forcer la cible en int 0/1 ───────────────────────────────────────────
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    return df


def build_dice_explainer(model, transformer, train_dataset: pd.DataFrame):
    """
    Construit et retourne l'objet DiCE `exp`.
    """
    train_cols = [c for c in train_dataset.columns if c != 'rejection']
    col_order  = joblib.load("col_order.pkl")

    def _transform(data, **kwargs):
        """
        Transforme les données pour le modèle.
        Ajoute les colonnes manquantes (ex: lost_MAT, final_creatinine…)
        attendues par le transformer mais absentes du dataset DiCE.
        """
        data = data.copy()
        for c in col_order:
            if c not in data.columns:
                data[c] = 0
        return transformer.transform(data[col_order])

    actual_continuous = [c for c in CONTINUOUS_FEATURES if c in train_dataset.columns]

    d = dice_ml.Data(
        dataframe=train_dataset,
        continuous_features=actual_continuous,
        outcome_name='rejection'
    )

    m = dice_ml.Model(
        model=model,
        backend='sklearn',
        func=_transform
    )

    return dice_ml.Dice(d, m)


def run_dice(exp, patient_row: pd.DataFrame, group: int = 1, total_cfs: int = 3):
    """
    Génère les counterfactuals pour un patient.
    """
    group_map = {1: FEATURES_GROUP1, 2: FEATURES_GROUP2, 3: FEATURES_GROUP3}
    features_to_vary = group_map.get(group, FEATURES_GROUP1)

    patient_row = patient_row.drop(columns=["rejection"], errors="ignore")

    # ── Traduire les valeurs incompatibles (race, induction…) ───────────────
    patient_row = _translate_for_dice(patient_row)

    # ── Aligner sur les colonnes exactes attendues par DiCE ─────────────────
    expected_cols = list(exp.data_interface.feature_names)

    extra_cols = [c for c in patient_row.columns if c not in expected_cols]
    if extra_cols:
        patient_row = patient_row.drop(columns=extra_cols)

    for c in [c for c in expected_cols if c not in patient_row.columns]:
        patient_row[c] = 0

    patient_row = patient_row[expected_cols]

    # ── Aligner les types ────────────────────────────────────────────────────
    for col in patient_row.columns:
        if col in CONTINUOUS_FEATURES:
            patient_row[col] = pd.to_numeric(patient_row[col], errors="coerce")
        else:
            try:
                patient_row[col] = patient_row[col].astype(int)
            except (ValueError, TypeError):
                patient_row[col] = patient_row[col].astype(str)

    # ── Validation catégorielles → affiche ❌ dans le terminal ───────────────
    _validate_categorical_values(patient_row, exp)

    # ── Vérifier les features disponibles dans le groupe choisi ─────────────
    available = [f for f in features_to_vary if f in patient_row.columns]

    if not available:
        raise ValueError(
            f"Aucune feature du groupe {group} trouvée dans patient_row.\n"
            f"Features attendues : {features_to_vary}\n"
            f"Colonnes disponibles : {list(patient_row.columns)}"
        )
    print("Colonnes patient_row final :", list(patient_row.columns))
    print("Colonnes attendues DiCE    :", expected_cols)
    dice_exp = exp.generate_counterfactuals(
        query_instances=patient_row,
        total_CFs=total_cfs,
        desired_class="opposite",
        features_to_vary=available,
    )

    return dice_exp.cf_examples_list[0].final_cfs_df