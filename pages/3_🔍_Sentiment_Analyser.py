"""Module 3 — Sentiment Analyser + Web Scraping"""
import streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import requests, re, time
from bs4 import BeautifulSoup
from transformers import pipeline
import warnings; warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sentiment Analyser | AI Nexus", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.mod-header { background:linear-gradient(135deg,#831843,#EC4899);
  padding:1.5rem 2rem; border-radius:12px; color:white; margin-bottom:1.5rem; }
.mod-header h2 { margin:0; font-size:1.5rem; font-weight:700; }
.mod-header p  { margin:0.3rem 0 0; opacity:0.8; font-size:0.88rem; }
.pos  { background:#D1FAE5; color:#065F46; padding:4px 12px; border-radius:20px; font-size:0.82rem; font-weight:600; }
.neg  { background:#FEE2E2; color:#7F1D1D; padding:4px 12px; border-radius:20px; font-size:0.82rem; font-weight:600; }
.neu  { background:#FEF3C7; color:#78350F; padding:4px 12px; border-radius:20px; font-size:0.82rem; font-weight:600; }
.rev-card { background:white; border:0.5px solid #FCE7F3; border-radius:10px;
  padding:0.85rem 1rem; margin-bottom:8px; border-left:4px solid #EC4899; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mod-header">
  <h2>🔍 Module 3 — Sentiment Analyser + Web Scraping</h2>
  <p>Scrape reviews or paste text → Hugging Face transformer classifies sentiment → visualise trends</p>
  <p><strong>Stack:</strong> BeautifulSoup · Hugging Face Transformers (DistilBERT) · NLTK · Seaborn · Python</p>
</div>
""", unsafe_allow_html=True)

# Cache the sentiment model
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    truncation=True, max_length=512)

def scrape_quotes(n=30):
    """Scrape quotes.toscrape.com as a demo scraping source (public, legal)."""
    reviews, tags_list = [], []
    try:
        for page in range(1, 4):
            r = requests.get(f"http://quotes.toscrape.com/page/{page}/", timeout=8)
            soup = BeautifulSoup(r.text, "html.parser")
            for q in soup.select(".quote"):
                text = q.select_one(".text").get_text(strip=True).strip('""')
                tag  = [t.get_text() for t in q.select(".tag")]
                reviews.append(text[:250])
                tags_list.append(", ".join(tag[:3]) if tag else "general")
            if len(reviews) >= n: break
    except Exception as e:
        st.warning(f"Scraping fallback — using sample data. ({e})")
    return reviews[:n], tags_list[:n]

def clean_text(text):
    text = re.sub(r'[^\w\s.,!?]', ' ', str(text))
    return re.sub(r'\s+', ' ', text).strip()

# ── Session ───────────────────────────────────────────────────────────────────
if "sent_df" not in st.session_state: st.session_state.sent_df = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Module 3 Setup")
    mode = st.radio("Input mode", ["🌐 Scrape live data","📝 Paste text manually","📂 Upload CSV"])
    if mode == "🌐 Scrape live data":
        n_reviews = st.slider("Reviews to scrape", 10, 50, 25)
        if st.button("🕷️ Scrape Now", use_container_width=True):
            with st.spinner("Scraping quotes.toscrape.com..."):
                reviews, tags = scrape_quotes(n_reviews)
                st.session_state.sent_df = pd.DataFrame({"text": reviews, "category": tags})
                st.success(f"✅ Scraped {len(reviews)} entries")
    elif mode == "📝 Paste text manually":
        raw = st.text_area("Paste reviews (one per line)", height=200,
                            placeholder="This product is amazing!\nTerrible quality, waste of money.\nDecent but could be better.")
        if raw and st.button("Analyse", use_container_width=True):
            lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
            st.session_state.sent_df = pd.DataFrame({"text": lines, "category": ["manual"]*len(lines)})
            st.success(f"✅ {len(lines)} texts ready")
    else:
        f = st.file_uploader("CSV with 'text' column", type=["csv"])
        if f:
            df_up = pd.read_csv(f)
            if "text" in df_up.columns:
                st.session_state.sent_df = df_up[["text"]].dropna()
                st.session_state.sent_df["category"] = "uploaded"
                st.success(f"✅ {len(st.session_state.sent_df)} rows loaded")
            else:
                st.error("CSV must have a 'text' column")

if st.session_state.sent_df is None:
    st.info("👈 Choose an input mode from the sidebar to begin.")
    st.markdown("""
    **How this works:**
    1. **Scrape** — BeautifulSoup extracts text from a public website
    2. **Clean** — regex removes noise from raw text
    3. **Classify** — DistilBERT (fine-tuned on SST-2) scores each text POSITIVE/NEGATIVE
    4. **Visualise** — Seaborn charts show distribution, trends, and category breakdown
    """)
    st.stop()

df_raw = st.session_state.sent_df.copy()
df_raw["text"] = df_raw["text"].apply(clean_text)
df_raw = df_raw[df_raw["text"].str.len() > 10].reset_index(drop=True)

# ── Run sentiment ─────────────────────────────────────────────────────────────
if st.button("🤖 Run Sentiment Analysis", use_container_width=True):
    with st.spinner("Loading DistilBERT and classifying..."):
        try:
            model = load_sentiment_model()
            results = model(df_raw["text"].tolist())
            df_raw["sentiment"] = [r["label"] for r in results]
            df_raw["confidence"] = [round(r["score"]*100, 2) for r in results]
            # Map to 3-class
            df_raw["sentiment_label"] = df_raw["sentiment"].map(
                {"POSITIVE":"Positive","NEGATIVE":"Negative"}).fillna("Neutral")
            st.session_state.sent_results = df_raw
            st.success(f"✅ Classified {len(df_raw)} texts | Avg confidence: {df_raw['confidence'].mean():.1f}%")
        except Exception as e:
            st.error(f"Model error: {e}")

if "sent_results" not in st.session_state: st.stop()
df = st.session_state.sent_results.copy()

# ── Metrics ───────────────────────────────────────────────────────────────────
pos = (df["sentiment_label"]=="Positive").sum()
neg = (df["sentiment_label"]=="Negative").sum()
avg_conf = df["confidence"].mean()
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Analysed", len(df))
c2.metric("Positive 😊", pos, f"{pos/len(df)*100:.1f}%")
c3.metric("Negative 😞", neg, f"{neg/len(df)*100:.1f}%")
c4.metric("Avg Confidence", f"{avg_conf:.1f}%")

tab1, tab2, tab3 = st.tabs(["📊 Charts", "📋 Results Table", "🔬 Deep Dive"])

with tab1:
    a,b = st.columns(2)
    with a:
        fig,ax = plt.subplots(figsize=(5,4))
        vc = df["sentiment_label"].value_counts()
        colors_s = {"Positive":"#10B981","Negative":"#EF4444","Neutral":"#F59E0B"}
        bars = ax.bar(vc.index, vc.values,
                      color=[colors_s.get(l,"#6B7280") for l in vc.index], edgecolor="white", linewidth=1.5)
        for bar,val in zip(bars,vc.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title("Sentiment Distribution", fontweight="bold")
        ax.set_ylabel("Count"); sns.despine(); st.pyplot(fig); plt.close()
    with b:
        fig,ax = plt.subplots(figsize=(5,4))
        for sent, clr in [("Positive","#10B981"),("Negative","#EF4444")]:
            sub = df[df["sentiment_label"]==sent]["confidence"]
            if len(sub): sns.kdeplot(sub, ax=ax, label=sent, color=clr, fill=True, alpha=0.3)
        ax.set_title("Confidence Distribution by Sentiment", fontweight="bold")
        ax.set_xlabel("Confidence (%)"); ax.legend(); sns.despine(); st.pyplot(fig); plt.close()

    if df["category"].nunique() > 1:
        fig,ax = plt.subplots(figsize=(9,4))
        cross = df.groupby(["category","sentiment_label"]).size().unstack(fill_value=0)
        cross.plot(kind="bar", ax=ax,
                   color=[colors_s.get(c,"#6B7280") for c in cross.columns],
                   edgecolor="white", linewidth=1.2)
        ax.set_title("Sentiment by Category", fontweight="bold")
        ax.set_xlabel(""); plt.xticks(rotation=30); ax.legend(title="Sentiment")
        sns.despine(); st.pyplot(fig); plt.close()

with tab2:
    filter_sent = st.multiselect("Filter by sentiment",
                                  df["sentiment_label"].unique().tolist(),
                                  default=df["sentiment_label"].unique().tolist())
    df_show = df[df["sentiment_label"].isin(filter_sent)][["text","sentiment_label","confidence","category"]]
    st.dataframe(df_show.rename(columns={"text":"Text","sentiment_label":"Sentiment",
                                          "confidence":"Confidence (%)","category":"Category"}),
                 use_container_width=True)
    csv = df_show.to_csv(index=False)
    st.download_button("📥 Download results CSV", csv, "sentiment_results.csv", "text/csv")

with tab3:
    st.markdown("#### Most Positive & Negative Texts")
    a,b = st.columns(2)
    with a:
        st.markdown("**Top 5 Positive**")
        top_pos = df[df["sentiment_label"]=="Positive"].nlargest(5,"confidence")
        for _,row in top_pos.iterrows():
            st.markdown(f"""<div class="rev-card">
            <span class="pos">Positive {row['confidence']:.1f}%</span>
            <p style="margin:6px 0 0;font-size:0.83rem;color:#374151">{row['text'][:180]}</p>
            </div>""", unsafe_allow_html=True)
    with b:
        st.markdown("**Top 5 Negative**")
        top_neg = df[df["sentiment_label"]=="Negative"].nlargest(5,"confidence")
        for _,row in top_neg.iterrows():
            st.markdown(f"""<div class="rev-card" style="border-left-color:#EF4444">
            <span class="neg">Negative {row['confidence']:.1f}%</span>
            <p style="margin:6px 0 0;font-size:0.83rem;color:#374151">{row['text'][:180]}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("#### Confidence Heatmap")
    if len(df) >= 10:
        fig,ax = plt.subplots(figsize=(10,3))
        pivot_data = df["confidence"].values[:50].reshape(1,-1)
        colors_conf = ["#EF4444" if df["sentiment_label"].iloc[i]=="Negative" else "#10B981"
                       for i in range(min(50,len(df)))]
        ax.bar(range(len(colors_conf)), pivot_data[0,:len(colors_conf)],
               color=colors_conf, width=0.85)
        ax.set_title("Confidence Score per Text (Red=Negative, Green=Positive)", fontweight="bold")
        ax.set_xlabel("Text index"); ax.set_ylabel("Confidence (%)")
        sns.despine(); st.pyplot(fig); plt.close()
