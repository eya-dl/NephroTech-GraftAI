"""
ui/chat_widget.py
Bouton FAB → st.switch_page("pages/chatbot.py")
C'est la méthode officielle Streamlit pour naviguer entre pages.
"""
import streamlit as st


def render_chat_widget():
    """
    Affiche un bouton FAB flottant en bas à droite.
    Clic → navigue vers pages/chatbot.py via st.switch_page.
    """
    # ── Style du FAB ──────────────────────────────────────────────────────────
    st.markdown("""
<style>
/* Cible le dernier bouton de la page (notre FAB) */
section.main > div.block-container div:last-of-type > div:last-child button {
    position: fixed !important;
    bottom: 28px !important;
    right: 28px !important;
    width: 56px !important;
    height: 56px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #0e7bdf, #1a5cc8) !important;
    color: white !important;
    font-size: 24px !important;
    border: none !important;
    box-shadow: 0 4px 18px rgba(14,123,223,0.5) !important;
    z-index: 999999 !important;
    padding: 0 !important;
    line-height: 1 !important;
    transition: transform 0.18s !important;
}
section.main > div.block-container div:last-of-type > div:last-child button:hover {
    transform: scale(1.09) !important;
    box-shadow: 0 6px 24px rgba(14,123,223,0.6) !important;
}

/* Badge vert animé via pseudo-élément */
section.main > div.block-container div:last-of-type > div:last-child {
    position: fixed !important;
    bottom: 28px !important;
    right: 28px !important;
    z-index: 999999 !important;
}
</style>
""", unsafe_allow_html=True)

    # ── Bouton natif Streamlit ────────────────────────────────────────────────
    if st.button("🩺", key="fab_chat_btn", help="Ouvrir MedAssist Chat"):
        st.switch_page("pages/chatbot.py")