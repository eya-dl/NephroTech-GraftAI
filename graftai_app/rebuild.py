import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ── Charger les données ──────────────────────────────────
# ✅ CORRIGÉ : read_excel au lieu de read_csv (fichier .xlsx)
data = pd.read_excel(
    r"C:\Users\eyadh\Desktop\PCD testing\LES VERSIONS EN COPIES\PROTOTYPE AVEC XAI\graftai_app\train_dataset.xlsx"
)

# ── Traduction ───────────────────────────────────────────
def traduction(df):
    code = {
        'SIM':'yes','sim':'yes','nao':'no','nào':'no','NAO':'no',
        'indeterminada':'indeterminate','outros':'other',
        'conservador':'conservative','falecido':'deceased','vivo':'alive',
        'CerebroVascular':'CerebroVascular','identico':0,'DISTINTO':6,
        'distinto':'different','haploidentico':3,'timo':'thymogllubulin',
        'tratamento':'treatment','profilatico':'prophylactic',
        'choque séptico':'septic shock','choque septico':'septic shock',
        'choque hemorragico/septico':'hemorrhagic/septic/shock',
        'choque séptico + hemorragico':'septic shock + hemorrhagic shock',
        '-':'no information','doacao':'donation','judical':'judicial',
        'nao recebeu':'did not receive','sem mutação':'no mutation',
        'não identificada':'not identified','nao realizada':'not carried out',
        'ausencia':'absence','dialise':np.nan,
    }
    final_creatinine_code = code.copy()
    final_creatinine_code['HD'] = np.nan
    final_creatinine_code['DP'] = np.nan
    for col in df.select_dtypes('object').columns:
        if col == 'final_creatinine':
            df.loc[:, col] = df[col].replace(final_creatinine_code)
        else:
            df.loc[:, col] = df[col].replace(code)
    if 'final_creatinine' in df.columns:
        df['final_creatinine'] = (
            df['final_creatinine']
            .astype(str)
            .str.replace(',', '.', regex=False)
        )
        df['final_creatinine'] = pd.to_numeric(
            df['final_creatinine'], errors='coerce'
        )
    return df


def drop_columns(df):
    cols_to_drop = [
        'lost_MAT', 'date.tx', 'ohter_information.Tx', 'cause_convertion',
        'date.diag', 'date_eculizumab', 'time_eculizumabe', 'variant',
        'ADAMST13', 'Shigatox', 'biopsy', 'trat.suport', 'change.IS',
        'time.trat.ecu', 'intercorrence.trat', 'date_event', 'variant.1',
        'genetic_description', 'final_creatinine', 'access_ecu', 'NA',
    ]
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)   # ← axis=1 supprimé


def nan_values(df):
    df.loc[df['sex'] == 'M', 'n_pregnant'] = 0
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
    df.loc[df['death'] == 'no', 'cause.death'] = 'still alive'
    df['MM'] = df['MM'].astype(int)
    return df


# ── Pipeline ─────────────────────────────────────────────
print("📂 Chargement des données...")
df = traduction(data)
df = drop_columns(df)
df = nan_values(df)

print(f"✅ Shape après preprocessing : {df.shape}")
print(f"📋 Colonnes : {df.columns.tolist()}")

X = df.drop('rejection', axis=1)
y = df['rejection'].replace({'yes': 1, 'no': 0}).astype(int)

print(f"\n🎯 Distribution cible :\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_cols = X_train.select_dtypes(
    include=['object', 'category']
).columns.tolist()

numerical_cols = X_train.select_dtypes(
    include='number'
).columns.tolist()

print(f"\n📊 Colonnes catégorielles ({len(categorical_cols)}) : {categorical_cols}")
print(f"📊 Colonnes numériques    ({len(numerical_cols)})   : {numerical_cols}")

# ── Transformer ──────────────────────────────────────────
# ✅ sparse_output=False est le bon paramètre pour sklearn >= 1.2
transformer = ColumnTransformer(transformers=[
    ('ohe',    OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
    ('scaler', StandardScaler(), numerical_cols)
])

X_train_t = transformer.fit_transform(X_train)
X_test_t  = transformer.transform(X_test)

print(f"\n✅ X_train transformé : {X_train_t.shape}")

# ── Modèle XGBoost ───────────────────────────────────────
print("\n🚀 Entraînement du modèle XGBoost...")
model_xgb = XGBClassifier(
    colsample_bytree=0.7,
    learning_rate=0.01,
    max_depth=3,
    n_estimators=250,
    scale_pos_weight=2.333,
    subsample=0.7,
    random_state=42,
    eval_metric='logloss',
)
model_xgb.fit(X_train_t, y_train)

# ── Évaluation rapide ────────────────────────────────────
from sklearn.metrics import classification_report, roc_auc_score
y_pred  = model_xgb.predict(X_test_t)
y_proba = model_xgb.predict_proba(X_test_t)[:, 1]
print("\n📈 Rapport de classification :")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC : {roc_auc_score(y_test, y_proba):.4f}")

# ── Sauvegarde ───────────────────────────────────────────
print("\n💾 Sauvegarde des fichiers .pkl...")

col_order = categorical_cols + numerical_cols   # ordre exact du transformer

col_order = categorical_cols + numerical_cols

joblib.dump(transformer,  "transformer.pkl")
joblib.dump(model_xgb,    "model.pkl")
joblib.dump(X_train_t,    "X_train.pkl")
joblib.dump(y_train,      "y_train.pkl")
joblib.dump(col_order,    "col_order.pkl")
joblib.dump(X_train,      "X_train_raw.pkl")  # ← ajoute cette ligne
print("✅ X_train_raw.pkl sauvegardé")

print("✅ Fichiers sauvegardés :")
print("   - transformer.pkl")
print("   - model.pkl")
print("   - X_train.pkl")
print("   - y_train.pkl")
print("   - col_order.pkl")
print(f"\n✅ Rebuild terminé avec succès sous Python {__import__('sys').version}!")
