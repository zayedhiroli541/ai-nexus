"""Module 4 — Customer Churn Predictor: Statistics + ML + AI Strategy (Groq – dynamic models)"""
import streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from groq import Groq
import requests, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Churn Predictor | AI Nexus", page_icon="🔮", layout="wide")

st.markdown("""
<style>
.mod-header { background:linear-gradient(135deg,#312E81,#6D28D9);
  padding:1.5rem 2rem; border-radius:12px; color:white; margin-bottom:1.5rem; }
.mod-header h2 { margin:0; font-size:1.5rem; font-weight:700; }
.mod-header p  { margin:0.3rem 0 0; opacity:0.8; font-size:0.88rem; }
.model-card { background:white; border:0.5px solid #EDE9FE; border-radius:10px;
  padding:0.9rem 1.1rem; text-align:center; }
.model-card .score { font-size:1.5rem; font-weight:700; color:#6D28D9; }
.model-card .name  { font-size:0.75rem; color:#6B7280; margin-top:2px; }
.model-card.best   { border:2px solid #6D28D9; }
.ai-strat { background:#F5F3FF; border:1px solid #C4B5FD; border-radius:10px;
  padding:1rem 1.25rem; font-size:0.88rem; color:#2E1065; line-height:1.75; white-space:pre-wrap; }
.stat-badge { padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.stat-sig  { background:#D1FAE5; color:#065F46; }
.stat-not  { background:#FEF3C7; color:#78350F; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mod-header">
  <h2>🔮 Module 4 — Customer Churn Predictor</h2>
  <p>5-algorithm comparison · CLT simulation · Distribution fitting · AI retention strategy</p>
  <p><strong>Stack:</strong> Pandas · Scikit‑Learn · SciPy · Matplotlib · Seaborn · Groq Llama‑3</p>
</div>
""", unsafe_allow_html=True)

# Session
for k, v in {"churn_df": None, "churn_groq_key": None, "churn_results": None, "churn_model": "llama-3.1-8b-instant"}.items():
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
        {"role": "system", "content": "You are a Customer Success Director. Be specific and business-focused."},
        {"role": "user", "content": prompt}
    ]
    last_error = None
    for model in model_list:
        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0.5, max_tokens=700)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            if "model_decommissioned" in str(e).lower() or "404" in str(e) or "400" in str(e):
                continue
            else: raise e
    raise last_error if last_error else RuntimeError("No model worked")


@st.cache_data
def telco_data():
    np.random.seed(42); n=1000
    return pd.DataFrame({
        "tenure": np.random.randint(1,72,n),
        "monthly_charges": np.random.uniform(20,120,n).round(2),
        "total_charges": (np.random.randint(1,72,n)*np.random.uniform(20,120,n)).round(2),
        "num_services": np.random.randint(1,8,n),
        "support_calls": np.random.randint(0,10,n),
        "contract": np.random.choice(["Month-to-month","One year","Two year"],n,p=[0.55,0.25,0.20]),
        "payment": np.random.choice(["Electronic check","Mailed check","Bank transfer","Credit card"],n),
        "satisfaction_score": np.random.randint(1,6,n),
        "churn": np.random.choice([0,1],n,p=[0.73,0.27])
    })


# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Module 4 Setup")
    groq_key = st.text_input("Groq API Key (AI strategy)", value=st.secrets.get("GROQ_API_KEY", ""),
                             type="password", placeholder="gsk_...")
    if groq_key:
        st.session_state.churn_groq_key = groq_key
        st.success("✅ Groq ready")
        available_models = fetch_available_groq_models(groq_key)
        if st.session_state.churn_model not in available_models:
            st.session_state.churn_model = available_models[0]
        st.markdown("**🤖 Model**")
        model_choice = st.selectbox("Choose model", options=available_models,
                                    index=available_models.index(st.session_state.churn_model)
                                    if st.session_state.churn_model in available_models else 0,
                                    help="Models fetched live from Groq. Auto-fallback on errors.")
        st.session_state.churn_model = model_choice
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

    src = st.radio("Data", ["Built-in Telco dataset", "Upload CSV"])
    if src == "Built-in Telco dataset":
        if st.button("Load Telco Data", use_container_width=True):
            st.session_state.churn_df = telco_data()
            st.success("✅ 1000 customers loaded")
    else:
        f = st.file_uploader("CSV with 'churn' column", type=["csv"])
        if f: st.session_state.churn_df = pd.read_csv(f); st.success("✅ Loaded")

if st.session_state.churn_df is None:
    st.info("👈 Load the Telco dataset from the sidebar.")
    st.stop()

df = st.session_state.churn_df.copy()
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

df_enc = df.copy()
for col in cat_cols:
    if col != "churn": df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

target = "churn" if "churn" in df.columns else df.columns[-1]
feat_cols = [c for c in df_enc.select_dtypes(include=np.number).columns if c != target]

tab1,tab2,tab3,tab4 = st.tabs(["📐 Stats Explorer", "🤖 5-Model Comparison", "📊 Churn Insights", "🧠 AI Strategy"])

# ── TAB 1: Statistical Explorer ──────────────────────────────────────────────
with tab1:
    st.markdown("#### Distribution Analysis & CLT Demo")
    num_sel = st.selectbox("Select feature", [c for c in num_cols if c!=target])
    colA,colB = st.columns(2)
    with colA:
        fig,ax = plt.subplots(figsize=(5,3.5))
        churned = df[df[target]==1][num_sel].dropna()
        retained = df[df[target]==0][num_sel].dropna()
        sns.kdeplot(retained, ax=ax, label="Retained", color="#10B981", fill=True, alpha=0.35)
        sns.kdeplot(churned, ax=ax, label="Churned", color="#EF4444", fill=True, alpha=0.35)
        ax.set_title(f"{num_sel} — Churn vs Retained", fontweight="bold"); ax.legend(); sns.despine(); st.pyplot(fig); plt.close()
    with colB:
        data = df[num_sel].dropna()
        fig,ax = plt.subplots(figsize=(5,3.5))
        ax.hist(data,bins=25,density=True,alpha=0.5,color="#6D28D9",edgecolor="white",label="Data")
        x=np.linspace(data.min(),data.max(),200)
        for d,nm,clr in [(stats.norm,"Normal","#EF4444"),(stats.expon,"Exponential","#F59E0B")]:
            try: ax.plot(x,d.pdf(x,*d.fit(data)),color=clr,linewidth=2,label=nm)
            except: pass
        ax.legend(fontsize=8); ax.set_title(f"Distribution Fitting — {num_sel}",fontweight="bold"); sns.despine(); st.pyplot(fig); plt.close()

    st.markdown("#### CLT Simulation")
    fig,axes = plt.subplots(1,4,figsize=(14,3))
    for ax,n in zip(axes,[10,30,50,200]):
        means=[df[num_sel].dropna().sample(n,replace=True).mean() for _ in range(1000)]
        ax.hist(means,bins=30,color="#6D28D9",alpha=0.75,edgecolor="white")
        ax.set_title(f"n={n}",fontweight="bold"); sns.despine(ax=ax)
    fig.suptitle(f"CLT — '{num_sel}' Sampling Distribution of Mean",fontweight="bold"); plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### T-Test: Churned vs Retained")
    ts,pv = stats.ttest_ind(df[df[target]==1][num_sel].dropna(), df[df[target]==0][num_sel].dropna())
    badge = "stat-sig" if pv<0.05 else "stat-not"
    st.markdown(f"t={ts:.3f} | p={pv:.5f} <span class='stat-badge {badge}'>{'Significant difference ✅' if pv<0.05 else 'No significant difference ⚠️'}</span>", unsafe_allow_html=True)

# ── TAB 2: 5-Model Comparison ────────────────────────────────────────────────
with tab2:
    if st.button("🚀 Train & Compare All 5 Models", use_container_width=True):
        X = df_enc[feat_cols].fillna(0); y = df_enc[target].fillna(0).astype(int)
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100,random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100,random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000,random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=6,random_state=42),
            "SVM": SVC(probability=True,random_state=42)
        }
        results={}
        with st.spinner("Training 5 models..."):
            for name,m in models.items():
                m.fit(Xtr_s,ytr); ypred=m.predict(Xte_s); yprob=m.predict_proba(Xte_s)[:,1]
                results[name]={"acc":round(accuracy_score(yte,ypred)*100,2),"auc":round(roc_auc_score(yte,yprob),4),"model":m,"yprob":yprob,"ypred":ypred}
        st.session_state.churn_results=results; st.session_state.churn_yte=yte; st.session_state.churn_feat=feat_cols

    if st.session_state.churn_results:
        res=st.session_state.churn_results; best=max(res,key=lambda k:res[k]["auc"])
        cols=st.columns(5)
        for col,(name,r) in zip(cols,res.items()):
            with col:
                st.markdown(f"<div class='model-card {'best' if name==best else ''}'><div class='score'>{r['auc']:.3f}</div><div class='name'>{name}<br>Acc: {r['acc']}%<br><b>{'🏆 Best' if name==best else ''}</b></div></div>", unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,5))
        for (name,r),c in zip(res.items(),["#6D28D9","#EC4899","#10B981","#F59E0B","#3B82F6"]):
            fpr,tpr,_=roc_curve(st.session_state.churn_yte,r["yprob"])
            ax.plot(fpr,tpr,label=f"{name} (AUC={r['auc']:.3f})",color=c,linewidth=2)
        ax.plot([0,1],[0,1],"k--"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curves",fontweight="bold"); ax.legend(); sns.despine(); st.pyplot(fig); plt.close()

        best_m=res[best]["model"]
        if hasattr(best_m,"feature_importances_"):
            imp=pd.DataFrame({"Feature":st.session_state.churn_feat,"Importance":best_m.feature_importances_}).sort_values("Importance")
            fig,ax=plt.subplots(figsize=(7,max(3,len(imp)*0.45)))
            ax.barh(imp["Feature"],imp["Importance"],color="#6D28D9"); ax.set_title(f"Feature Importance — {best}",fontweight="bold"); sns.despine(); st.pyplot(fig); plt.close()

# ── TAB 3: Churn Insights ────────────────────────────────────────────────────
with tab3:
    churn_rate = df[target].mean()*100 if df[target].dtype in [int,float] else 0
    st.metric("Overall Churn Rate",f"{churn_rate:.1f}%")
    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(5,4))
        ax.pie([df[df[target]==0].shape[0],df[df[target]==1].shape[0]],labels=["Retained","Churned"],colors=["#10B981","#EF4444"],autopct="%1.1f%%",startangle=90); ax.set_title("Churn vs Retained",fontweight="bold"); st.pyplot(fig); plt.close()
    with c2:
        if "monthly_charges" in df.columns:
            fig,ax=plt.subplots(figsize=(5,4))
            df.groupby(target)["monthly_charges"].mean().plot(kind="bar",color=["#10B981","#EF4444"],ax=ax,edgecolor="white")
            ax.set_title("Avg Monthly Charges by Churn",fontweight="bold"); sns.despine(); st.pyplot(fig); plt.close()
    if cat_cols:
        cat_sel=st.selectbox("Churn breakdown by category",cat_cols)
        ct=pd.crosstab(df[cat_sel],df[target],normalize="index")*100
        fig,ax=plt.subplots(figsize=(8,4)); ct.plot(kind="bar",ax=ax,color=["#10B981","#EF4444"],edgecolor="white",linewidth=1.2)
        ax.set_title(f"Churn Rate by {cat_sel}",fontweight="bold"); plt.xticks(rotation=30); sns.despine(); st.pyplot(fig); plt.close()

# ── TAB 4: AI Strategy ────────────────────────────────────────────────────────
with tab4:
    if not st.session_state.churn_groq_key:
        st.warning("👈 Add your Groq API key in the sidebar for AI strategy.")
    else:
        key=st.session_state.churn_groq_key
        model=st.session_state.churn_model
        stats_str=df[num_cols[:5]].describe().round(2).to_string() if num_cols else ""
        churn_pct=f"{df[target].mean()*100:.1f}%" if df[target].dtype in [int,float] else "N/A"
        priority_models=[model]+[m for m in FALLBACK_MODELS if m!=model]

        st.markdown("#### Single Customer Churn Risk Calculator")
        if st.session_state.churn_results:
            st.markdown("Enter a customer's values to predict their churn probability:")

            # We'll use only these 9 features for the demo risk model
            short_feats = feat_cols[:9]
            inp = {}

            cols_inp = st.columns(3)
            for i, feat in enumerate(short_feats):
                with cols_inp[i % 3]:
                    mn = float(df_enc[feat].min())
                    mx = float(df_enc[feat].max())
                    inp[feat] = st.slider(feat, mn, mx,
                                          float(df_enc[feat].median()),
                                          key=f"risk_{feat}")

            if st.button("Calculate Risk"):
                # ── Train a simple model on the same 9 features ──────────────
                # This guarantees the number of features always matches the sliders,
                # regardless of how many columns are in the full dataset.
                X_short = df_enc[short_feats].fillna(0)
                y_short = df_enc[target].fillna(0).astype(int)

                # Use the same test split for a fair evaluation (optional)
                Xtr_s, _, ytr_s, _ = train_test_split(
                    X_short, y_short, test_size=0.2, random_state=42
                )
                sc2 = StandardScaler()
                Xtr_s_scaled = sc2.fit_transform(Xtr_s)

                # Train a fast, reliable model
                risk_model = LogisticRegression(max_iter=1000, random_state=42)
                risk_model.fit(Xtr_s_scaled, ytr_s)

                # ── Predict for the single customer ─────────────────────────
                X_inp = np.array([[inp[f] for f in short_feats]])
                X_inp_scaled = sc2.transform(X_inp)
                prob = risk_model.predict_proba(X_inp_scaled)[0][1] * 100

                color = "#EF4444" if prob > 50 else "#10B981"
                risk = "HIGH RISK 🔴" if prob > 50 else "LOW RISK 🟢"

                st.markdown(f"""
                <div style="background:{color}15;border:2px solid {color};
                border-radius:10px;padding:1rem;text-align:center;margin-top:1rem;">
                <div style="font-size:2rem;font-weight:800;color:{color}">{prob:.1f}%</div>
                <div style="font-weight:600;color:{color}">{risk}</div>
                <div style="font-size:0.82rem;color:#374151;margin-top:4px">
                Churn Probability (Logistic Regression on 9 features)
                </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Train models first in the 5-Model Comparison tab to use the risk calculator.")