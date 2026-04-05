"""
pages/chatbot.py
Page dédiée MedAssist Chat — Transplantation Rénale.
Accessible via le bouton FAB de l'app principale.
"""
import time
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="MedAssist Chat",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "timestamps" not in st.session_state: st.session_state.timestamps = []
if "sources"    not in st.session_state: st.session_state.sources    = []

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.stApp { background: #eef1f6; }
.block-container { padding: 0 !important; max-width: 680px !important; margin: 0 auto; }

/* ── Header ── */
.cw-header {
    background: linear-gradient(135deg, #0e7bdf 0%, #1a5cc8 100%);
    padding: 13px 16px 11px;
    display: flex; align-items: center; gap: 10px;
    position: sticky; top: 0; z-index: 100;
}
.cw-back {
    background: rgba(255,255,255,0.15); border: none;
    color: #fff; border-radius: 8px;
    padding: 4px 10px; font-size: 13px; cursor: pointer;
    text-decoration: none; display: flex; align-items: center; gap: 4px;
}
.cw-back:hover { background: rgba(255,255,255,0.25); }
.cw-avatar {
    width: 38px; height: 38px; border-radius: 50%;
    background: rgba(255,255,255,0.22);
    border: 2px solid rgba(255,255,255,0.55);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.cw-title { font-size: 14px; font-weight: 600; color: #fff; }
.cw-status {
    font-size: 11px; color: rgba(255,255,255,0.82);
    display: flex; align-items: center; gap: 4px; margin-top: 1px;
}
.cw-dot {
    width: 6px; height: 6px; background: #4ade80;
    border-radius: 50%; animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }

.cw-sub {
    background: #e8f1fb; padding: 7px 16px;
    font-size: 11px; color: #185fa5; font-weight: 500;
    text-align: center; border-bottom: 0.5px solid #b5d4f4;
}

/* ── Messages ── */
.cw-date-sep { text-align: center; margin: 10px 0 14px; }
.cw-date-sep span {
    background: #dde3ed; color: #6b7a99;
    font-size: 10.5px; padding: 3px 10px;
    border-radius: 10px; font-weight: 500;
}
.cw-row { display: flex; align-items: flex-end; gap: 7px; margin-bottom: 4px; padding: 0 8px; animation: popIn 0.18s ease; }
.cw-row.u { flex-direction: row-reverse; }
@keyframes popIn {
    from { opacity:0; transform:scale(0.96) translateY(4px); }
    to   { opacity:1; transform:scale(1) translateY(0); }
}
.cw-av {
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; flex-shrink: 0; margin-bottom: 2px;
}
.cw-av.b { background: linear-gradient(135deg,#0e7bdf,#1a5cc8); }
.cw-av.u { background: #dde3ed; }
.cw-av.h { visibility: hidden; }
.cw-bw { display: flex; flex-direction: column; max-width: 72%; }
.cw-bw.u { align-items: flex-end; }
.cw-bubble {
    padding: 9px 13px; border-radius: 16px;
    font-size: 13px; line-height: 1.55; word-wrap: break-word;
}
.cw-bubble.b {
    background: #fff; color: #1c1e21;
    border-bottom-left-radius: 4px;
    border: 0.5px solid #dde1ea;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.cw-bubble.u {
    background: #0e7bdf; color: #fff;
    border-bottom-right-radius: 4px;
    box-shadow: 0 2px 6px rgba(14,123,223,0.28);
}
.cw-time { font-size: 10px; color: #9aa3b5; margin-top: 2px; padding: 0 2px; }
.cw-src-badge {
    display: inline-flex; align-items: center; gap: 3px;
    background: #eef2fb; border: 0.5px solid #b5d4f4;
    border-radius: 10px; padding: 2px 7px;
    font-size: 10.5px; color: #185fa5; margin: 2px 2px 0;
}
.cw-rating { display: flex; align-items: center; gap: 5px; margin-top: 3px; }
.cw-rating span { font-size: 10.5px; color: #9aa3b5; }
.cw-rbtn {
    background: #fff; border: 0.5px solid #dde1ea;
    border-radius: 50%; width: 22px; height: 22px;
    display: inline-flex; align-items: center; justify-content: center; font-size: 11px;
}

/* ── Typing ── */
.cw-typing { display: flex; align-items: flex-end; gap: 7px; padding: 0 8px; }
.cw-tbubble {
    background: #fff; border: 0.5px solid #dde1ea;
    border-radius: 16px; border-bottom-left-radius: 4px;
    padding: 10px 14px; display: flex; gap: 4px; align-items: center;
}
.cw-td { width: 6px; height: 6px; background: #adb5c7; border-radius: 50%; animation: ta 1.2s infinite; }
.cw-td:nth-child(2){animation-delay:.2s} .cw-td:nth-child(3){animation-delay:.4s}
@keyframes ta { 0%,60%,100%{transform:translateY(0);opacity:.4} 30%{transform:translateY(-5px);opacity:1} }

/* ── Input ── */
[data-testid="stChatInput"] { background: #fff !important; border-top: 0.5px solid #dde1ea !important; padding: 8px 10px !important; }
[data-testid="stChatInput"] textarea {
    background: #f0f4fa !important; border: 0.5px solid #dde1ea !important;
    border-radius: 22px !important; padding: 9px 16px !important;
    color: #1c1e21 !important; font-family: 'DM Sans', sans-serif !important; font-size: 13px !important;
}
[data-testid="stChatInput"] textarea:focus { border-color: #0e7bdf !important; box-shadow: none !important; }
[data-testid="stChatInput"] button { background: #0e7bdf !important; border-radius: 50% !important; }

/* ── Clear button ── */
.clear-btn > button {
    background: #fff !important; border: 0.5px solid #dde1ea !important;
    color: #9aa3b5 !important; border-radius: 7px !important;
    font-size: 11px !important; padding: 2px 8px !important;
}
.cw-powered {
    text-align: center; font-size: 10px; color: #b0b8c9;
    padding: 6px 0 4px; background: #fff;
    border-top: 0.5px solid #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cw-header">
    <a class="cw-back" href="/" title="Retour à GraftAI">← Retour</a>
    <div class="cw-avatar">🩺</div>
    <div>
        <div class="cw-title">MedAssist — Transplantation Rénale</div>
        <div class="cw-status"><span class="cw-dot"></span> En ligne · API connectée</div>
    </div>
</div>
<div class="cw-sub">Nous répondons généralement en quelques secondes.</div>
""", unsafe_allow_html=True)

# ── Bouton effacer ─────────────────────────────────────────────────────────────
_, col_clear = st.columns([6, 1])
with col_clear:
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️", key="clear"):
        st.session_state.messages   = []
        st.session_state.timestamps = []
        st.session_state.sources    = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Date ───────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div class='cw-date-sep'><span>Aujourd'hui · {time.strftime('%d %b %Y')}</span></div>",
    unsafe_allow_html=True
)

# ── Messages ───────────────────────────────────────────────────────────────────
def render_messages():
    prev_role = None
    for i, msg in enumerate(st.session_state.messages):
        role    = msg["role"]
        content = msg["content"]
        ts      = st.session_state.timestamps[i] if i < len(st.session_state.timestamps) else ""
        srcs    = st.session_state.sources[i]    if i < len(st.session_state.sources)    else []

        hide   = "h" if role == prev_role else ""
        row_c  = "u" if role == "user" else ""
        av_c   = "u" if role == "user" else "b"
        bub_c  = "u" if role == "user" else "b"
        bw_c   = "u" if role == "user" else ""
        icon   = "👤" if role == "user" else "🩺"

        src_html = ""
        if srcs and role == "assistant":
            badges   = "".join(f'<span class="cw-src-badge">📄 {s}</span>' for s in srcs)
            src_html = f"<div style='margin-top:3px'>{badges}</div>"

        rating_html = ""
        if role == "assistant":
            rating_html = """
            <div class="cw-rating">
                <span>Utile ?</span>
                <span class="cw-rbtn">👍</span>
                <span class="cw-rbtn">👎</span>
            </div>"""

        st.markdown(f"""
        <div class="cw-row {row_c}">
            <div class="cw-av {av_c} {hide}">{icon}</div>
            <div class="cw-bw {bw_c}">
                <div class="cw-bubble {bub_c}">{content}</div>
                {src_html}{rating_html}
                <div class="cw-time">{ts}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        prev_role = role

if not st.session_state.messages:
    st.markdown("""
    <div class="cw-row">
        <div class="cw-av b">🩺</div>
        <div class="cw-bw">
            <div class="cw-bubble b">
                Bonjour ! Je suis MedAssist, votre assistant spécialisé en transplantation rénale.
                Posez-moi vos questions sur les traitements, les facteurs de risque de rejet,
                les biomarqueurs ou l'interprétation des scores XAI.
            </div>
            <div class="cw-time">maintenant</div>
        </div>
    </div>""", unsafe_allow_html=True)
else:
    render_messages()

# Placeholder typing
typing_placeholder = st.empty()

st.markdown("<div class='cw-powered'>PROPULSÉ PAR MedAssist AI</div>", unsafe_allow_html=True)

# ── Traitement message en attente ──────────────────────────────────────────────
if st.session_state.get("pending_msg"):
    prompt = st.session_state.pending_msg
    st.session_state.pending_msg = None

    typing_placeholder.markdown("""
    <div class="cw-typing">
        <div class="cw-av b">🩺</div>
        <div class="cw-tbubble">
            <div class="cw-td"></div><div class="cw-td"></div><div class="cw-td"></div>
        </div>
    </div>""", unsafe_allow_html=True)

    try:
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        response = requests.post(
            f"{API_URL}/chat",
            json={"question": prompt, "history": history},
            timeout=30,
        )
        response.raise_for_status()
        data    = response.json()
        answer  = data["answer"]
        sources = data["sources"]

        typing_placeholder.empty()
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.timestamps.append(time.strftime("%H:%M"))
        st.session_state.sources.append(sources)

    except requests.exceptions.ConnectionError:
        typing_placeholder.empty()
        st.error("❌ API injoignable. Lance : `uvicorn api:app --reload`")
    except requests.exceptions.Timeout:
        typing_placeholder.empty()
        st.error("⏱️ Délai dépassé.")
    except requests.exceptions.HTTPError as e:
        typing_placeholder.empty()
        st.error(f"❌ Erreur HTTP {e.response.status_code}")
    except Exception as e:
        typing_placeholder.empty()
        st.error(f"Erreur : {e}")

    st.rerun()

# ── Saisie ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.timestamps.append(time.strftime("%H:%M"))
    st.session_state.sources.append([])
    st.session_state.pending_msg = prompt
    st.rerun()