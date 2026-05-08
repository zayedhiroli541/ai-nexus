"""Module 2 — AI Data Insights: EDA + Statistics + GenAI Narration (Groq – dynamic models)"""
import streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from groq import Groq
import requests, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Data Insights | AI Nexus", page_icon="📊", layout="wide")

st.markdown("""
<style>
.mod-header { background:linear-gradient(135deg,#065F46,#10B981);
  padding:1.5rem 2rem; border-radius:12px; color:white; margin-bottom:1.5rem; }
.mod-header h2 { margin:0; font-size:1.5rem; font-weight:700; }
.mod-header p  { margin:0.3rem 0 0; opacity:0.8; font-size:0.88rem; }
.kpi { background:#F0FDF4; border:1px solid #A7F3D0; border-radius:10px; padding:0.9rem; text-align:center; }
.kpi .val { font-size:1.6rem; font-weight:700; color:#065F46; }
.kpi .lbl { font-size:0.7rem; color:#6B7280; }
.ai-box { background:#F0FDF9; border:1px solid #99F6E4; border-radius:10px;
  padding:1rem 1.25rem; font-size:0.88rem; color:#134E4A; line-height:1.75; white-space:pre-wrap; }
.stat-ok { background:#DCFCE7; color:#14532D; padding:3px 10px; border-radius:20px; font-size:0.8rem; }
.stat-no { background:#FEE2E2; color:#7F1D1D; padding:3px 10px; border-radius:20px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mod-header">
  <h2>📊 Module 2 — AI Data Insights Engine</h2>
  <p>EDA · Statistical Tests (CLT, Normality, T-test) · ML Predictions · Clustering · AI Narrative</p>
  <p><strong>Stack:</strong> Pandas · NumPy · SciPy · Scikit‑Learn · Matplotlib · Seaborn · Groq Llama‑3</p>
</div>
""", unsafe_allow_html=True)

# Session
for k, v in {"ins_df": None, "ins_groq_key": None, "ins_model": "llama-3.1-8b-instant"}.items():
    if k not in st.session_state: st.session_state[k] = v

# ── Dynamic model fetching ─────────────────────────────────────────────────
FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

@st.cache_data(ttl=86400)
def fetch_available_groq_models(api_key: str):
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            available = []
            for m in data.get("data", []):
                if not m.get("active"): continue
                if "chat" not in m.get("type", ""): continue
                if "preview" in m.get("id", "").lower(): continue
                available.append(m["id"])
            if available: return available
    except: pass
    return FALLBACK_MODELS


def ask_groq(api_key: str, prompt: str, model_list: list) -> str:
    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a senior data scientist and business analyst. Be specific, data-driven, and concise."},
        {"role": "user", "content": prompt}
    ]
    last_error = None
    for model in model_list:
        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0.4, max_tokens=600)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            if "model_decommissioned" in str(e).lower() or "404" in str(e) or "400" in str(e):
                continue
            else: raise e
    raise last_error if last_error else RuntimeError("No model worked")


@st.cache_data
def samples():
    np.random.seed(42); n=300
    churn = pd.DataFrame({
        "age": np.random.randint(18,70,n), "tenure": np.random.randint(1,72,n),
        "monthly_charges": np.random.uniform(20,120,n).round(2),
        "support_calls": np.random.randint(0,10,n),
        "satisfaction": np.random.randint(1,6,n),
        "churn": np.random.choice([0,1],n,p=[0.73,0.27])
    })
    sales = pd.DataFrame({
        "month": np.tile(range(1,13),25), "units_sold": np.random.randint(50,500,n),
        "price": np.random.uniform(10,200,n).round(2),
        "marketing": np.random.uniform(500,5000,n).round(2),
        "revenue": (np.random.randint(50,500,n)*np.random.uniform(10,200,n)).round(2)
    })
    return {"Customer Churn": churn, "Sales Analytics": sales}


# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Module 2 Setup")
    groq_key = st.text_input("Groq API Key (AI narration)", value=st.secrets.get("GROQ_API_KEY", ""),
                             type="password", placeholder="gsk_...")
    if groq_key:
        st.session_state.ins_groq_key = groq_key
        st.success("✅ Groq ready")
        available_models = fetch_available_groq_models(groq_key)
        if st.session_state.ins_model not in available_models:
            st.session_state.ins_model = available_models[0]
        st.markdown("**🤖 Model**")
        model_choice = st.selectbox("Choose model", options=available_models,
                                    index=available_models.index(st.session_state.ins_model)
                                    if st.session_state.ins_model in available_models else 0,
                                    help="Models fetched live from Groq. Auto-fallback on errors.")
        st.session_state.ins_model = model_choice
    else:
        st.info(
         "🔑 **Need a Groq API key?**\n\n"
         "1. Go to [console.groq.com](https://console.groq.com) — **sign up free** (Google/GitHub)\n"
         "2. Click **API Keys** → **Create API Key**\n"
         "3. Copy the key (starts with `gsk_`)\n"
         "4. Paste it in the input above\n\n"
         "⚠️ No credit card required. Free for 30 requests/min, 14,400/day.\n"
         "Your key is never stored by this app."
        )

    src = st.radio("Data Source", ["Sample dataset", "Upload CSV"])
    if src == "Sample dataset":
        sel = st.selectbox("Choose", list(samples().keys()))
        if st.button("Load", use_container_width=True):
            st.session_state.ins_df = samples()[sel].copy()
            st.success(f"✅ Loaded {sel}")
    else:
        f = st.file_uploader("CSV", type=["csv"])
        if f: st.session_state.ins_df = pd.read_csv(f); st.success("✅ Loaded")

if st.session_state.ins_df is None:
    st.info("👈 Load a dataset from the sidebar.")
    st.stop()

df = st.session_state.ins_df.copy()
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# KPIs
c1,c2,c3,c4 = st.columns(4)
for col,val,lbl in zip([c1,c2,c3,c4],
    [df.shape[0], df.shape[1], len(num_cols), int(df.isnull().sum().sum())],
    ["Rows","Columns","Numeric Cols","Missing Values"]):
    with col: st.markdown(f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 EDA", "📐 Statistics", "🤖 ML Pipeline", "🧩 Clustering", "✨ AI Insights"])

# ── TAB 1: EDA ────────────────────────────────────────────────────────────────
with tab1:
    st.dataframe(df.describe().round(3), use_container_width=True)
    if len(num_cols) >= 2:
        col = st.selectbox("Explore column", num_cols)
        a,b = st.columns(2)
        with a:
            fig,ax = plt.subplots(figsize=(5,3.5))
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#10B981", alpha=0.7)
            ax.set_title(f"Distribution — {col}", fontweight="bold"); sns.despine(); st.pyplot(fig); plt.close()
        with b:
            fig,ax = plt.subplots(figsize=(5,3.5))
            stats.probplot(df[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot — {col}", fontweight="bold")
            ax.get_lines()[0].set(color="#10B981",markersize=3); ax.get_lines()[1].set(color="#EF4444",linewidth=1.5)
            sns.despine(); st.pyplot(fig); plt.close()
        fig,ax = plt.subplots(figsize=(8,5))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="Greens", ax=ax, linewidths=0.5, annot_kws={"size":9})
        ax.set_title("Correlation Heatmap", fontweight="bold"); st.pyplot(fig); plt.close()

# ── TAB 2: Statistics ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Shapiro-Wilk Normality Tests")
    rows=[]
    for c in num_cols:
        s = df[c].dropna().sample(min(200,len(df[c].dropna())),random_state=42)
        stat,p = stats.shapiro(s)
        rows.append({"Column":c,"W-Stat":round(stat,4),"p-value":round(p,5),
                     "Normal?":"✅ Yes" if p>0.05 else "❌ No"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("#### CLT Demonstration")
    clt_c = st.selectbox("Column", num_cols, key="clt")
    fig,axes = plt.subplots(1,4,figsize=(14,3.5))
    for ax,n in zip(axes,[10,30,50,100]):
        means=[df[clt_c].dropna().sample(n,replace=True).mean() for _ in range(1000)]
        ax.hist(means,bins=30,color="#10B981",alpha=0.75,edgecolor="white")
        ax.set_title(f"n={n}",fontweight="bold"); sns.despine(ax=ax)
    fig.suptitle(f"CLT Demo — '{clt_c}' (1000 iterations)",fontweight="bold"); plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Distribution Fitting")
    dc = st.selectbox("Column", num_cols, key="dist2")
    data = df[dc].dropna()
    fig,ax = plt.subplots(figsize=(8,4))
    ax.hist(data,bins=30,density=True,alpha=0.5,color="#10B981",edgecolor="white",label="Data")
    x=np.linspace(data.min(),data.max(),200)
    for d,nm,clr in [(stats.norm,"Normal","#EF4444"),(stats.lognorm,"Log-Normal","#3B82F6"),(stats.expon,"Exponential","#F59E0B")]:
        try: ax.plot(x,d.pdf(x,*d.fit(data)),color=clr,linewidth=2,label=nm)
        except: pass
    ax.legend(); ax.set_title(f"Distribution Fitting — {dc}",fontweight="bold"); sns.despine(); st.pyplot(fig); plt.close()

    if cat_cols:
        st.markdown("#### Two-Sample T-Test")
        tc = st.selectbox("Group by", cat_cols, key="tg")
        tn = st.selectbox("Test variable", num_cols, key="tv")
        grps = df[tc].dropna().unique()
        if len(grps)>=2:
            g1=df[df[tc]==grps[0]][tn].dropna(); g2=df[df[tc]==grps[1]][tn].dropna()
            t,p = stats.ttest_ind(g1,g2)
            color = "stat-ok" if p<0.05 else "stat-no"
            st.markdown(f'<span class="{color}">t={t:.4f} | p={p:.5f} | {"Significant ✅" if p<0.05 else "Not significant ⚠️"}</span>', unsafe_allow_html=True)

# ── TAB 3: ML ─────────────────────────────────────────────────────────────────
with tab3:
    target = st.selectbox("Target column", df.columns.tolist(), index=len(df.columns)-1)
    feats  = st.multiselect("Feature columns", num_cols, default=[c for c in num_cols if c!=target][:5])
    if feats and st.button("🚀 Train Model", use_container_width=True):
        X=df[feats].dropna(); y=df.loc[X.index,target].dropna(); X=X.loc[y.index]
        is_clf = y.dtype=="object" or y.nunique()<=10
        if is_clf and y.dtype=="object": y=pd.Series(LabelEncoder().fit_transform(y),index=y.index)
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
        if is_clf:
            m=RandomForestClassifier(n_estimators=100,random_state=42).fit(Xtr_s,ytr); ypred=m.predict(Xte_s)
            st.success(f"✅ Accuracy: {accuracy_score(yte,ypred)*100:.2f}%")
        else:
            m=RandomForestRegressor(n_estimators=100,random_state=42).fit(Xtr_s,ytr); ypred=m.predict(Xte_s)
            c1,c2=st.columns(2); c1.metric("R²",round(r2_score(yte,ypred),4)); c2.metric("RMSE",round(np.sqrt(mean_squared_error(yte,ypred)),4))
        imp=pd.DataFrame({"Feature":feats,"Importance":m.feature_importances_}).sort_values("Importance")
        fig,ax=plt.subplots(figsize=(7,max(3,len(feats)*0.5)))
        ax.barh(imp["Feature"],imp["Importance"],color="#10B981"); ax.set_title("Feature Importance",fontweight="bold"); sns.despine(); st.pyplot(fig); plt.close()

# ── TAB 4: Clustering (FIXED – works on first click) ──────────────────────────
with tab4:
    cf = st.multiselect("Clustering features", num_cols, default=num_cols[:min(4,len(num_cols))])
    k  = st.slider("K clusters",2,8,3)

    # Session state flag to remember the button click
    if "run_clustering" not in st.session_state:
        st.session_state.run_clustering = False

    if st.button("🔍 Cluster + PCA", use_container_width=True):
        st.session_state.run_clustering = True   # set the flag

    # If flag is True and features are selected, run the clustering
    if st.session_state.run_clustering and cf:
        Xc=df[cf].dropna(); Xs=StandardScaler().fit_transform(Xc)
        inertias=[KMeans(n_clusters=ki,random_state=42,n_init=10).fit(Xs).inertia_ for ki in range(2,9)]
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        axes[0].plot(range(2,9),inertias,"bo-",color="#10B981"); axes[0].axvline(k,color="red",linestyle="--")
        axes[0].set_title("Elbow Method",fontweight="bold"); sns.despine(ax=axes[0])
        labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(Xs)
        X2d=PCA(n_components=2).fit_transform(Xs)
        colors=["#10B981","#EF4444","#3B82F6","#F59E0B","#8B5CF6","#EC4899","#06B6D4","#84CC16"]
        for i in range(k): axes[1].scatter(X2d[labels==i,0],X2d[labels==i,1],alpha=0.65,s=30,color=colors[i],label=f"Cluster {i+1}")
        axes[1].set_title(f"KMeans K={k} (PCA 2D)",fontweight="bold"); axes[1].legend(fontsize=8)
        sns.despine(ax=axes[1]); plt.tight_layout(); st.pyplot(fig); plt.close()
        Xc2=Xc.copy(); Xc2["Cluster"]=labels
        st.dataframe(Xc2.groupby("Cluster").mean().round(2), use_container_width=True)

        # Reset the flag after generating charts so they stay visible
        # We only reset if the button is clicked again later
        st.session_state.run_clustering = False

# ── TAB 5: AI Insights ────────────────────────────────────────────────────────
with tab5:
    if not st.session_state.ins_groq_key:
        st.warning("👈 Add your Groq API key in the sidebar to enable AI narration.")
    else:
        key = st.session_state.ins_groq_key
        model = st.session_state.ins_model
        desc = df[num_cols].describe().round(2).to_string() if num_cols else ""
        cols_str = ", ".join(df.columns[:12].tolist())
        shape_str = f"{df.shape[0]} rows × {df.shape[1]} cols"

        priority_models = [model] + [m for m in FALLBACK_MODELS if m != model]

        st.markdown("#### EDA Summary")
        if st.button("Generate EDA insights"):
            with st.spinner("Groq Llama-3 thinking..."):
                try:
                    r = ask_groq(key, f"Dataset: {shape_str}. Columns: {cols_str}.\nStats:\n{desc[:600]}\nWrite 4 bullet points (start with •) highlighting key patterns.", model_list=priority_models)
                    st.markdown(f'<div class="ai-box">{r}</div>', unsafe_allow_html=True)
                except Exception as e: st.error(f"❌ {e}")

        st.markdown("#### Business Recommendations")
        biz = st.text_input("Business context (optional)")
        if st.button("Generate recommendations"):
            with st.spinner("Groq Llama-3 thinking..."):
                try:
                    ctx = f"Context: {biz}. " if biz else ""
                    r = ask_groq(key, f"{ctx}Dataset: {shape_str}. Columns: {cols_str}. Provide 4 numbered actionable recommendations with expected impact.", model_list=priority_models)
                    st.markdown(f'<div class="ai-box">{r}</div>', unsafe_allow_html=True)
                except Exception as e: st.error(f"❌ {e}")

        st.markdown("#### Ask a Custom Question")
        q = st.text_area("Your question")
        if q and st.button("Ask AI"):
            with st.spinner("Groq Llama-3 thinking..."):
                try:
                    r = ask_groq(key, f"Expert data analyst. Dataset: {shape_str}. Columns: {cols_str}. Stats: {desc[:400]}. Question: {q}. Answer concisely with data-backed reasoning.", model_list=priority_models)
                    st.markdown(f'<div class="ai-box">{r}</div>', unsafe_allow_html=True)
                except Exception as e: st.error(f"❌ {e}")