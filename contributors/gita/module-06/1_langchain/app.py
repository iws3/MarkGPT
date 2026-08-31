import streamlit as st

st.set_page_config(
    page_title="Pep AI — Tactical Analyst",
    page_icon=":brain:",
    layout="wide",
)

PEP_IMG = "https://i.ibb.co/hFmwnbBF/Chat-GPT-Image-Aug-31-2026-04-02-31-PM-removebg-preview.png"

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at 30% 25%, #1a1a2e 0%, #0a0a0f 55%, #000000 100%);
    }

    .block-container {
        max-width: 1100px;
        padding-top: 3rem;
    }

    .hero-row {
        display: flex;
        align-items: center;
        gap: 3rem;
    }

    .glow-behind {
        position: relative;
        flex: 0 0 340px;
        display: flex;
        justify-content: center;
    }

    .glow-behind::before {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 420px;
        height: 420px;
        background: radial-gradient(circle, rgba(0,150,255,0.35) 0%, rgba(255,140,0,0.18) 45%, transparent 75%);
        filter: blur(45px);
        z-index: 0;
    }

    .glow-behind img {
        position: relative;
        z-index: 1;
        width: 100%;
        max-height: 560px;
        object-fit: contain;
        filter: drop-shadow(0 0 25px rgba(0,150,255,0.25));
    }

    .eyebrow {
        letter-spacing: 4px;
        text-transform: uppercase;
        font-size: 0.75rem;
        color: #8a8a9e;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: 3.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff 0%, #a8c5ff 50%, #ffb347 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 1.1rem 0;
        line-height: 1.1;
    }

    .hero-sub {
        color: #b8b8c8;
        font-size: 1.05rem;
        max-width: 460px;
        margin-bottom: 1.8rem;
        line-height: 1.6;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #0096ff, #ff8c00);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.75rem 2.2rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 0 25px rgba(0,150,255,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 0 35px rgba(255,140,0,0.45);
        color: white;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.2rem;
        margin-top: 3.5rem;
    }

    .feature-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.4rem;
        backdrop-filter: blur(6px);
    }

    .feature-card h4 {
        color: #e8e8f0;
        margin: 0 0 0.4rem 0;
        font-size: 1rem;
    }

    .feature-card p {
        color: #9a9ab0;
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero-row">
    <div class="glow-behind">
        <img src="{PEP_IMG}" alt="Pep AI">
    </div>
    <div>
        <div class="eyebrow">Tactical Intelligence, On Demand</div>
        <div class="hero-title">Talk Tactics<br>with Pep AI</div>
        <div class="hero-sub">
            Formations, pressing triggers, build-up play, player roles —
            broken down with the precision of a manager who never stops thinking about the game.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

_, btn_col = st.columns([0.38, 0.62])
with btn_col:
    if st.button("Start the Session"):
        st.switch_page("pages/1_Chat.py")

st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <h4>Formation Breakdowns</h4>
        <p>4-3-3, inverted fullbacks, false nines — explained structurally, not superficially.</p>
    </div>
    <div class="feature-card">
        <h4>Pressing & Transitions</h4>
        <p>Understand triggers, lines of engagement, and how possession flips in seconds.</p>
    </div>
    <div class="feature-card">
        <h4>Coaching Plans</h4>
        <p>Session structures and periodization concepts, grounded in real tactical logic.</p>
    </div>
</div>
""", unsafe_allow_html=True)