"""
╔══════════════════════════════════════════════════════════════════╗
║           AI NEXUS — All-in-One AI & Data Science Platform       ║
║  Built by: Zayed Asif Hiroli                                     ║
║  Stack   : Python · LangChain · FAISS · HuggingFace · Scikit-   ║
║            Learn · SciPy · Pandas · NumPy · Seaborn · FastAPI ·  ║
║            BeautifulSoup · Streamlit · Groq (free Llama‑3)       ║
║  Covers  : GenAI · RAG · NLP · MLOps · Data Science · Statistics ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st

st.set_page_config(
    page_title="AI Nexus | Zayed Asif Hiroli",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

/* Hero */
.nexus-hero {
    background: linear-gradient(135deg, #080F2E 0%, #0F1B4C 40%, #1B4F9B 80%, #2563EB 100%);
    padding: 3rem 2.5rem; border-radius: 18px; color: white; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.nexus-hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 240px; height: 240px; border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.nexus-hero::after {
    content: ''; position: absolute; bottom: -80px; left: 30%;
    width: 300px; height: 300px; border-radius: 50%;
    background: rgba(255,255,255,0.03);
}
.nexus-hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0 0 0.4rem; letter-spacing: -0.5px; }
.nexus-hero .sub { font-size: 1rem; opacity: 0.75; margin: 0 0 1.2rem; }
.nexus-hero .author { font-size: 0.82rem; opacity: 0.6; margin-top: 1.5rem; }

/* Module cards */
.module-card {
    background: white; border: 0.5px solid #E2E8F5;
    border-radius: 14px; padding: 1.4rem 1.5rem;
    transition: all 0.2s; cursor: pointer; height: 100%;
    position: relative; overflow: hidden;
}
.module-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 4px; height: 100%; border-radius: 14px 0 0 14px;
}
.module-card.c1::before { background: #6366F1; }
.module-card.c2::before { background: #10B981; }
.module-card.c3::before { background: #F59E0B; }
.module-card.c4::before { background: #EC4899; }
.module-card.c5::before { background: #3B82F6; }

.module-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.module-title { font-size: 1rem; font-weight: 700; color: #0F1B4C; margin-bottom: 0.3rem; }
.module-desc  { font-size: 0.82rem; color: #64748B; line-height: 1.55; margin-bottom: 0.8rem; }

.tech-pill { display: inline-block; font-size: 0.7rem; padding: 2px 8px;
             border-radius: 20px; margin: 2px; font-weight: 500; }
.p1 { background: #EEF2FF; color: #4338CA; }
.p2 { background: #D1FAE5; color: #065F46; }
.p3 { background: #FEF3C7; color: #92400E; }
.p4 { background: #FCE7F3; color: #9D174D; }
.p5 { background: #DBEAFE; color: #1E40AF; }
.p6 { background: #F3E8FF; color: #6B21A8; }
.p7 { background: #FFE4E6; color: #9F1239; }
.p8 { background: #E0F2FE; color: #0C4A6E; }

/* Stats bar */
.stat-bar { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.5rem; }
.stat-chip { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
             border-radius: 20px; padding: 5px 14px; font-size: 0.78rem;
             color: white; font-weight: 500; }

/* Impact badges */
.impact { font-size: 0.72rem; font-weight: 600; padding: 3px 10px;
          border-radius: 20px; display: inline-block; margin-top: 4px; }
.impact-vh { background: #DCFCE7; color: #14532D; }
.impact-h  { background: #DBEAFE; color: #1E3A8A; }

/* Section heading */
.sec-head { font-size: 1.3rem; font-weight: 700; color: #0F1B4C;
            margin: 0.5rem 0 1rem; padding-bottom: 0.5rem;
            border-bottom: 2px solid #EEF2FF; }

/* Resume box */
.resume-box { background: #F8FAFF; border: 1px solid #C7D5F0;
              border-radius: 10px; padding: 1rem 1.25rem;
              font-size: 0.85rem; color: #1E293B; line-height: 1.7;
              margin-top: 0.5rem; font-style: italic; }

/* Nav hint & key banner */
.nav-hint { background: #EEF2FF; border-radius: 10px; padding: 1rem 1.25rem;
            font-size: 0.85rem; color: #4338CA; line-height: 1.6; }
.groq-badge { background: #F0FDF4; border: 1px solid #86EFAC; border-radius: 8px;
              padding: 0.5rem 1rem; font-size: 0.85rem; color: #14532D; margin-bottom: 1rem; }
.key-notice { background: #FFF7ED; border: 1px solid #FDBA74; border-radius: 8px;
              padding: 0.75rem 1rem; font-size: 0.85rem; color: #7C2D12; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Hero  ──────────────────────────────────────────────
st.markdown("""
<div class="nexus-hero">
  <h1>🧠 AI Nexus</h1>
  <p class="sub">One platform. Five domains. The complete AI & Data Science toolkit — dynamically fetching the newest free Groq LLMs, with automatic fallback so it never breaks.</p>
  <div class="stat-bar">
    <span class="stat-chip">🤖 RAG + LLMs (Groq)</span>
    <span class="stat-chip">📊 Data Science</span>
    <span class="stat-chip">📐 Statistics</span>
    <span class="stat-chip">🔍 NLP + Sentiment</span>
    <span class="stat-chip">⚙️ MLOps Pipeline</span>
    <span class="stat-chip">🌐 Web Scraping</span>
    <span class="stat-chip">🔮 Churn Prediction</span>
  </div>
  <p class="author">
    Built by <strong>Zayed Asif Hiroli</strong> · Oracle Certified GenAI Professional ·
    github.com/zayedhiroli541 · linkedin.com/in/zayed-hiroli-5a15762a6
  </p>
</div>
""", unsafe_allow_html=True)

# ── Clickable profile buttons ────────────────────────────────────────
col1, col2, _ = st.columns([0.3, 0.3, 0.4])
with col1:
    st.link_button("🐙 GitHub", "https://github.com/zayedhiroli541")
with col2:
    st.link_button("💼 LinkedIn", "https://www.linkedin.com/in/zayed-hiroli-5a15762a6")

    st.markdown("<br>", unsafe_allow_html=True)  


# Groq + key notice
st.markdown("""
<div class="groq-badge">
⚡ <b>Groq LLM API</b> — models are fetched <b>live</b> from Groq, only active free chat models appear.
If the selected model fails, a <b>built‑in fallback chain</b> automatically tries the next one – zero user errors.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="key-notice">
🔑 <b>Need an API key?</b> This app does <b>not</b> ship with a key. You can get a <b>free Groq API key</b> in 2 minutes at
<a href="https://console.groq.com" target="_blank">console.groq.com</a> (no credit card, 30 req/min, 14,400 req/day).
Paste it in the sidebar of any LLM‑powered module to unlock AI features.
</div>
""", unsafe_allow_html=True)

# ── Module cards ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">🗂️ Platform Modules — Use the sidebar to navigate</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="module-card c1">
      <div class="module-icon">📄</div>
      <div class="module-title">Module 1 — RAG PDF Chatbot</div>
      <div class="module-desc">Upload any PDF and ask questions in natural language.
      Powered by LangChain + FAISS vector search, and <b>Groq’s free LLMs</b> (dynamic list).
      <b>One‑click suggestions</b> now instantly generate answers. Automatic model fallback ensures zero downtime.</div>
      <div>
        <span class="tech-pill p1">LangChain</span>
        <span class="tech-pill p1">FAISS</span>
        <span class="tech-pill p8">Groq</span>
        <span class="tech-pill p2">RAG</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="module-card c2">
      <div class="module-icon">📊</div>
      <div class="module-title">Module 2 — AI Data Insights</div>
      <div class="module-desc">Upload any CSV dataset. Get full EDA, statistical tests
      (Shapiro‑Wilk, CLT, t‑test, distribution fitting), ML predictions,
      clustering + PCA <b>(now works on first click)</b>, and AI‑narrated reports via Groq with fallback.</div>
      <div>
        <span class="tech-pill p2">Pandas</span>
        <span class="tech-pill p3">SciPy</span>
        <span class="tech-pill p5">Scikit-Learn</span>
        <span class="tech-pill p8">Groq</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="module-card c3">
      <div class="module-icon">🔍</div>
      <div class="module-title">Module 3 — Sentiment Analyser</div>
      <div class="module-desc">Scrape live product reviews or paste text.
      Hugging Face transformer classifies sentiment with deep learning.
      Visualise trends across brands, time periods, and star ratings.</div>
      <div>
        <span class="tech-pill p6">Transformers</span>
        <span class="tech-pill p4">NLP</span>
        <span class="tech-pill p7">BeautifulSoup</span>
        <span class="tech-pill p3">Seaborn</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
c4, c5, _ = st.columns(3)

with c4:
    st.markdown("""
    <div class="module-card c4">
      <div class="module-icon">🔮</div>
      <div class="module-title">Module 4 — Churn Predictor</div>
      <div class="module-desc">Upload customer data or use built‑in Telco dataset.
      Compare 5 ML algorithms, run CLT simulation, distribution fitting,
      and get an <b>AI‑generated retention strategy</b> via Groq with fallback. <b>Risk calculator works after a single training run.</b></div>
      <div>
        <span class="tech-pill p5">Scikit-Learn</span>
        <span class="tech-pill p3">Statistics</span>
        <span class="tech-pill p2">EDA</span>
        <span class="tech-pill p8">Groq</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown("""
    <div class="module-card c5">
      <div class="module-icon">⚙️</div>
      <div class="module-title">Module 5 — MLOps Pipeline</div>
      <div class="module-desc">Train, evaluate, and deploy ML models in‑browser.
      Full pipeline: preprocessing → training → evaluation → live REST API
      simulation → Docker + GitHub Actions CI/CD. <b>Seamless – no extra clicks needed</b>.</div>
      <div>
        <span class="tech-pill p5">Scikit-Learn</span>
        <span class="tech-pill p7">FastAPI</span>
        <span class="tech-pill p3">Docker</span>
        <span class="tech-pill p2">CI/CD</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Nav hint ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="nav-hint">
  👈 <strong>Use the sidebar to navigate between modules.</strong>
  Each module is fully independent — you can jump to any one directly.
  <br>Start with <strong>Module 1 (RAG PDF Chatbot)</strong> to see GenAI in action,
  or <strong>Module 2 (AI Data Insights)</strong> for the full Data Science + Statistics experience.
  <br><b>All LLM‑powered modules let you choose any available Groq model on the fly — the list auto‑updates with live fetching and auto‑fallback.</b>
</div>
""", unsafe_allow_html=True)

# ── Modules bullets ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-head">📝 Modules Bullet Points (ready to use)</div>', unsafe_allow_html=True)

bullets = [
    ("Module 1 — RAG",
     "Built a RAG‑based PDF Q&A chatbot using LangChain, FAISS, and Hugging Face embeddings; integrated with Groq’s free LLM API featuring dynamic model fetching and automatic fallback, achieving sub‑2s answers from 50‑page documents; one‑click suggestion buttons trigger instant generation."),
    ("Module 2 — Data + GenAI",
     "Developed an AI‑powered statistical insights engine combining Pandas EDA, SciPy hypothesis testing (Shapiro‑Wilk, CLT, t‑test), Scikit‑Learn ML pipelines, KMeans clustering, PCA, and Groq‑generated business narratives with live model selection and fallback."),
    ("Module 3 — NLP",
     "Built a sentiment analysis pipeline scraping 500+ product reviews using BeautifulSoup, classified using a Hugging Face transformer model, visualised trends across brands and ratings using Seaborn."),
    ("Module 4 — Churn",
     "Developed a customer churn prediction system using 5 Scikit‑Learn algorithms with CLT simulation, distribution fitting, and AI‑generated retention strategy via Groq with dynamic models and fallback; includes a real‑time churn risk calculator."),
    ("Module 5 — MLOps",
     "Engineered a full MLOps pipeline: data ingestion → feature engineering → model training → REST API simulation via FastAPI → Docker containerisation guide with GitHub Actions CI/CD; all stages work seamlessly without extra user steps."),
    ("Overall (AI Generalist)",
     "Built AI Nexus — a unified 5‑module AI & Data Science platform covering RAG, NLP, statistical analysis, churn prediction, and MLOps deployment, powered by Groq’s free LLM API with dynamic model fetching & auto‑fallback for zero‑cost, resilient inference; deployed on Streamlit Cloud with live demo."),
]

for label, bullet in bullets:
    st.markdown(f"""
    <div style="margin-bottom:10px;">
      <div style="font-size:0.75rem;font-weight:600;color:#6366F1;margin-bottom:3px;">{label}</div>
      <div class="resume-box">"{bullet}"</div>
    </div>
    """, unsafe_allow_html=True)