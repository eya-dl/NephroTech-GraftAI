# ═══════════════════════════════════════════════════════
#  ui/dice_display.py — Affichage DiCE dans GraftAI
# ═══════════════════════════════════════════════════════
import pandas as pd
import streamlit as st


# ── Libellés affichés pour chaque groupe ────────────────────────────────────
GROUP_LABELS = {
    1: (
        "💊 Scénario 1 — Traitement post-greffe",
        "Seules les variables de traitement immunosuppresseur sont modifiées.",
    ),
    2: (
        "🏥 Scénario 2 — Donneur & Chirurgie",
        "Seules les variables liées au donneur et à la procédure chirurgicale sont modifiées.",
    ),
    3: (
        "🔀 Scénario 3 — Combiné (Traitement + Donneur)",
        "Toutes les variables modifiables sont prises en compte simultanément.",
    ),
}


def _style_cf_table(df: pd.DataFrame, patient_ref: pd.Series):
    """Applique un style coloré : vert si valeur modifiée, gris sinon."""
    def highlight(val, col):
        if str(val) != str(patient_ref.get(col, "")):
            return "background-color:#1a3a2a;color:#4ade80;font-weight:600"
        return "color:#6b7280"

    styled = df.style
    for col in df.columns:
        styled = styled.map(
            lambda v, c=col: highlight(v, c),
            subset=[col],
        )
    return (
        styled
        .set_properties(**{
            "text-align": "center",
            "border": "1px solid #2d3748",
            "padding": "6px 12px",
        })
        .set_table_styles([
            {"selector": "thead th",
             "props": [("background-color", "#0f2027"),
                       ("color", "#7dd3fc"),
                       ("font-size", "0.82rem"),
                       ("text-transform", "uppercase"),
                       ("letter-spacing", "0.05em"),
                       ("border-bottom", "2px solid #3b82f6")]},
            {"selector": "tbody tr:hover",
             "props": [("background-color", "#1e293b")]},
        ])
        .format(precision=3, na_rep="—")
    )


def render_dice_section(exp, input_df: pd.DataFrame) -> None:
    """
    Affiche la section DiCE (Counterfactual Explanations) dans l'interface.

    Parameters
    ----------
    exp      : dice_ml.Dice — explainer déjà construit (via build_dice_explainer)
    input_df : pd.DataFrame (1 ligne, col_order) — brut depuis render_patient_form()
               Même structure exacte que le dataset d'entraînement.
    """
    from dice_runner import run_dice

    st.divider()
    st.markdown("## 🔁 Explications Contrefactuelles — DiCE")
    st.markdown(
        "DiCE génère des **scénarios alternatifs** : quelles modifications minimales "
        "des variables permettraient d'**inverser la prédiction** de rejet ?"
    )

    # ── Paramètres utilisateur ───────────────────────────────────────────────
    col_a, col_b = st.columns([2, 1])
    with col_a:
        group = st.radio(
            "Scénario à explorer",
            options=[1, 2, 3],
            format_func=lambda g: GROUP_LABELS[g][0],
            horizontal=True,
            key="dice_group",
        )
    with col_b:
        total_cfs = st.slider(
            "Nombre de counterfactuals", min_value=1, max_value=5,
            value=3, key="dice_n_cfs",
        )

    title, description = GROUP_LABELS[group]
    st.caption(f"ℹ️ {description}")

    # ── Bouton de génération ─────────────────────────────────────────────────
    if st.button("⚡ Générer les counterfactuals", key="btn_dice"):
        with st.spinner("Calcul des scénarios contrefactuels en cours…"):
            try:
                cf_df = run_dice(exp, input_df, group=group, total_cfs=total_cfs)
                st.session_state["dice_result"]      = cf_df
                st.session_state["dice_input_df"]    = input_df.copy()
                st.session_state["dice_group_label"] = title
            except Exception as e:
                st.error(f"❌ Erreur DiCE : {e}")
                st.session_state.pop("dice_result", None)

    # ── Affichage du résultat ────────────────────────────────────────────────
    if "dice_result" in st.session_state:
        cf_df: pd.DataFrame  = st.session_state["dice_result"]
        ref_df: pd.DataFrame = st.session_state["dice_input_df"]
        label: str           = st.session_state["dice_group_label"]

        st.markdown(f"### {label}")
        st.markdown(
            f"**{len(cf_df)} scénario(s) généré(s)** — "
            "seules les colonnes **différentes du patient original** sont affichées."
        )

        # ── Filtrer uniquement les colonnes qui changent ─────────────────────
        common_cols = [c for c in cf_df.columns if c in ref_df.columns]
        patient_ref = ref_df[common_cols].iloc[0]

        display_df = cf_df[common_cols].copy()
        display_df.index = [f"Scénario {i+1}" for i in range(len(display_df))]

        changed_cols = [
            col for col in common_cols
            if not (display_df[col].astype(str) == str(patient_ref[col])).all()
        ]

        if changed_cols:
            st.dataframe(
                _style_cf_table(display_df[changed_cols], patient_ref),
                use_container_width=True,
            )

            with st.expander("👤 Valeurs originales du patient (colonnes modifiées)"):
                orig = pd.DataFrame(
                    [patient_ref[changed_cols].values],
                    columns=changed_cols,
                    index=["Patient original"],
                )
                st.dataframe(orig, use_container_width=True)

            st.markdown(
                "<small style='color:#6b7280;'>🟢 Valeur modifiée &nbsp;|&nbsp; "
                "Valeurs identiques masquées</small>",
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "Aucun changement détecté dans les colonnes communes. "
                "Essayez d'augmenter le nombre de counterfactuals ou de changer de scénario."
            )

        # ── Téléchargement CSV ───────────────────────────────────────────────
        csv = cf_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Télécharger les counterfactuals (CSV)",
            data=csv,
            file_name="counterfactuals.csv",
            mime="text/csv",
            key="dice_download",
        )