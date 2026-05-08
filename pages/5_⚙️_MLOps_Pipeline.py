"""Module 5 — MLOps Pipeline Simulator: Train → Evaluate → Deploy → Monitor"""
import streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import json, pickle, io, time, os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="MLOps Pipeline | AI Nexus", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
.mod-header { background:linear-gradient(135deg,#0C4A6E,#0284C7);
  padding:1.5rem 2rem; border-radius:12px; color:white; margin-bottom:1.5rem; }
.mod-header h2 { margin:0; font-size:1.5rem; font-weight:700; }
.mod-header p  { margin:0.3rem 0 0; opacity:0.8; font-size:0.88rem; }
.pipe-stage { background:white; border:0.5px solid #BAE6FD; border-radius:10px;
  padding:1rem 1.25rem; margin-bottom:1rem; border-left:4px solid #0284C7; }
.pipe-stage h4 { margin:0 0 0.5rem; color:#0C4A6E; font-size:0.95rem; }
.stage-done { background:#F0FDF4; border-left-color:#10B981; }
.stage-active { background:#F0F9FF; border-left-color:#0284C7; }
.metric-pill { background:#E0F2FE; color:#0C4A6E; padding:4px 12px;
  border-radius:20px; font-size:0.82rem; font-weight:600; margin:2px; display:inline-block; }
.code-box { background:#0F172A; color:#E2E8F0; padding:1rem 1.25rem; border-radius:10px;
  font-family:monospace; font-size:0.8rem; line-height:1.6; overflow-x:auto; white-space:pre; }
.api-box { background:#F0F9FF; border:1px solid #BAE6FD; border-radius:10px;
  padding:1rem 1.25rem; font-family:monospace; font-size:0.82rem; color:#0C4A6E; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mod-header">
  <h2>⚙️ Module 5 — MLOps Pipeline Simulator</h2>
  <p>Data Ingestion → Feature Engineering → Model Training → Evaluation → API Simulation → Docker Guide</p>
  <p><strong>Stack:</strong> Scikit-Learn · Pandas · FastAPI (simulated) · Docker · GitHub Actions · Python</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────
for k, v in {
    "mlops_df": None,
    "mlops_X": None,
    "mlops_y": None,
    "mlops_trained": False,
    "mlops_metrics": None,
    "mlops_model": None,
    "mlops_algo": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_data
def get_sample():
    np.random.seed(42)
    n = 800
    return pd.DataFrame({
        "age": np.random.randint(18, 65, n),
        "income": np.random.randint(20000, 120000, n),
        "credit_score": np.random.randint(300, 850, n),
        "loan_amount": np.random.randint(1000, 50000, n),
        "employment_years": np.random.randint(0, 30, n),
        "num_accounts": np.random.randint(1, 10, n),
        "missed_payments": np.random.randint(0, 5, n),
        "debt_to_income": np.random.uniform(0.1, 0.9, n).round(3),
        "loan_default": np.random.choice([0, 1], n, p=[0.80, 0.20]),
    })


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Module 5 Setup")

    # Data loading – only show load button if no data is present
    if st.session_state.mlops_df is None:
        src = st.radio("Dataset", ["Sample (Loan Default)", "Upload CSV"])
        if src == "Sample (Loan Default)":
            if st.button("📦 Load Sample Data", use_container_width=True):
                st.session_state.mlops_df = get_sample()
                st.session_state.mlops_X = None      # reset any previous features
                st.session_state.mlops_y = None
                st.session_state.mlops_trained = False
                st.rerun()   # immediately show the data
        else:
            f = st.file_uploader("CSV", type=["csv"])
            if f:
                st.session_state.mlops_df = pd.read_csv(f)
                st.session_state.mlops_X = None
                st.session_state.mlops_y = None
                st.session_state.mlops_trained = False
                st.rerun()
    else:
        st.success("✅ Data loaded")
        # Offer a way to start over (but keep it out of the way)
        if st.button("🔄 Reload / Change Data"):
            st.session_state.mlops_df = None
            st.session_state.mlops_X = None
            st.session_state.mlops_y = None
            st.session_state.mlops_trained = False
            st.rerun()

    # Target selection (only if data exists)
    if st.session_state.mlops_df is not None:
        df = st.session_state.mlops_df.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        num_col_names = [c for c in df.columns if df[c].dtype in [int, float]]
        target = st.selectbox(
            "Target column",
            num_col_names,
            index=len(num_col_names) - 1,
            key="mlops_target"
        )
        # Update feature list based on current target
        st.session_state.feat_cols = [c for c in num_cols if c != target]


# ── Main area ─────────────────────────────────────────────────────────────
if st.session_state.mlops_df is None:
    st.info("👈 Load data from the sidebar to begin the MLOps pipeline.")
    st.markdown("#### MLOps Pipeline Overview")
    stages = [
        ("1. Data Ingestion", "Load, validate, and profile raw data"),
        ("2. Feature Engineering", "Encoding, scaling, imputation, pipeline construction"),
        ("3. Model Training", "Train with cross-validation"),
        ("4. Evaluation", "Confusion matrix, ROC, feature importance"),
        ("5. Deployment", "API simulation, Docker, CI/CD"),
    ]
    for stage, desc in stages:
        st.markdown(f"""<div class="pipe-stage">
        <h4>⚙️ {stage}</h4>
        <p style="margin:0;font-size:0.82rem;color:#374151">{desc}</p>
        </div>""", unsafe_allow_html=True)
    st.stop()

# Data is loaded – get the current DataFrame and target
df = st.session_state.mlops_df.copy()
target = st.session_state.get("mlops_target", df.columns[-1])
num_cols = df.select_dtypes(include=np.number).columns.tolist()
feat_cols = st.session_state.get("feat_cols", [c for c in num_cols if c != target])

# ══ PIPELINE TABS ══
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Data Ingestion",
    "🔧 Feature Engineering",
    "🎯 Model Training",
    "📊 Evaluation",
    "🚀 Deploy",
])

# ── TAB 1: Ingestion ──────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="pipe-stage stage-done"><h4>✅ Stage 1 — Data Ingestion & Validation</h4></div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", int(df.isnull().sum().sum()))
    c4.metric("Numeric cols", len(num_cols))

    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("**Data Types & Nulls**")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isnull().sum(),
        "null_%": (df.isnull().sum() / len(df) * 100).round(2),
        "unique": df.nunique(),
    })
    st.dataframe(info, use_container_width=True)
    st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

# ── TAB 2: Feature Engineering ────────────────────────────────────────────
with tab2:
    st.markdown('<div class="pipe-stage stage-active"><h4>🔧 Stage 2 — Feature Engineering</h4></div>',
                unsafe_allow_html=True)
    apply_scale = st.checkbox("StandardScaler", value=True)
    apply_enc = st.checkbox("Encode categoricals (LabelEncoder)", value=True)
    apply_fillna = st.checkbox("Fill missing values (median)", value=True)

    if st.button("⚙️ Build Pipeline", use_container_width=True):
        df_proc = df.copy()
        if apply_fillna:
            df_proc = df_proc.fillna(df_proc.median(numeric_only=True))

        cat_c = [c for c in df_proc.select_dtypes(include="object").columns if c != target]
        if apply_enc:
            for c in cat_c:
                df_proc[c] = LabelEncoder().fit_transform(df_proc[c].astype(str))

        X = df_proc[feat_cols].fillna(0)
        y = df_proc[target].fillna(0).astype(int)

        if apply_scale:
            sc = StandardScaler()
            X_sc = pd.DataFrame(sc.fit_transform(X), columns=feat_cols)
        else:
            X_sc = X

        # Store prepared data
        st.session_state.mlops_X = X_sc
        st.session_state.mlops_y = y
        st.success(f"✅ Pipeline built — {X_sc.shape[0]} rows × {X_sc.shape[1]} features")
        st.dataframe(X_sc.head(5), use_container_width=True)

        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(X_sc.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    annot_kws={"size": 8}, linewidths=0.5)
        ax.set_title("Feature Correlation (post-engineering)", fontweight="bold")
        st.pyplot(fig)
        plt.close()

# ── TAB 3: Model Training ────────────────────────────────────────────────
with tab3:
    # Check if features are ready (pipeline built)
    if "mlops_X" not in st.session_state or st.session_state.mlops_X is None:
        st.info("Complete Feature Engineering first.")
        st.stop()

    st.markdown('<div class="pipe-stage stage-active"><h4>🎯 Stage 3 — Model Training</h4></div>',
                unsafe_allow_html=True)

    X = st.session_state.mlops_X
    y = st.session_state.mlops_y

    model_choice = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
    test_size = st.slider("Test split %", 10, 40, 20)
    cv_folds = st.slider("CV folds", 3, 10, 5)

    if st.button("🚀 Train Model", use_container_width=True):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size / 100, random_state=42)

        model_map = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        }
        m = model_map[model_choice]

        with st.spinner(f"Training {model_choice}..."):
            t0 = time.time()
            m.fit(Xtr, ytr)
            train_time = round(time.time() - t0, 3)
            ypred = m.predict(Xte)
            yprob = m.predict_proba(Xte)[:, 1]
            cv_scores = cross_val_score(m, X, y, cv=cv_folds, scoring="accuracy")

        metrics = {
            "accuracy": round(accuracy_score(yte, ypred) * 100, 2),
            "auc": round(roc_auc_score(yte, yprob), 4),
            "cv_mean": round(cv_scores.mean() * 100, 2),
            "cv_std": round(cv_scores.std() * 100, 2),
            "train_time": train_time,
            "train_size": len(Xtr),
            "test_size": len(Xte),
        }

        st.session_state.mlops_metrics = metrics
        st.session_state.mlops_model = m
        st.session_state.mlops_Xte = Xte
        st.session_state.mlops_yte = yte
        st.session_state.mlops_yprob = yprob
        st.session_state.mlops_ypred = ypred
        st.session_state.mlops_trained = True
        st.session_state.mlops_algo = model_choice

        st.success(f"✅ {model_choice} trained in {train_time}s")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Accuracy", f"{metrics['accuracy']}%")
        p2.metric("ROC-AUC", metrics['auc'])
        p3.metric(f"CV Mean ({cv_folds}-fold)", f"{metrics['cv_mean']}%")
        p4.metric("CV Std", f"±{metrics['cv_std']}%")

# ── TAB 4: Evaluation ─────────────────────────────────────────────────────
with tab4:
    if not st.session_state.get("mlops_trained", False):
        st.info("Train a model first.")
        st.stop()

    st.markdown('<div class="pipe-stage stage-done"><h4>📊 Stage 4 — Model Evaluation</h4></div>',
                unsafe_allow_html=True)

    m = st.session_state.mlops_metrics
    algo = st.session_state.mlops_algo
    pill_texts = [
        f"Algorithm: {algo}",
        f"Accuracy: {m['accuracy']}%",
        f"AUC: {m['auc']}",
        f"CV: {m['cv_mean']}±{m['cv_std']}%",
        f"Train time: {m['train_time']}s",
    ]
    for txt in pill_texts:
        st.markdown(f'<span class="metric-pill">{txt}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    yte = st.session_state.mlops_yte
    ypred = st.session_state.mlops_ypred
    yprob = st.session_state.mlops_yprob

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(yte, ypred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Predicted 0", "Predicted 1"],
                    yticklabels=["Actual 0", "Actual 1"])
        ax.set_title("Confusion Matrix", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with col2:
        fpr, tpr, _ = roc_curve(yte, yprob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#0284C7", linewidth=2.5, label=f"AUC={m['auc']}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curve", fontweight="bold")
        ax.legend()
        sns.despine()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Classification Report**")
    st.text(classification_report(yte, ypred))

    model_obj = st.session_state.mlops_model
    if hasattr(model_obj, "feature_importances_"):
        imp_df = pd.DataFrame({
            "Feature": feat_cols,
            "Importance": model_obj.feature_importances_
        }).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(7, max(3, len(imp_df) * 0.45)))
        ax.barh(imp_df["Feature"], imp_df["Importance"], color="#0284C7")
        ax.set_title("Feature Importance", fontweight="bold")
        sns.despine()
        st.pyplot(fig)
        plt.close()

    # Download model
    buf = io.BytesIO()
    pickle.dump(model_obj, buf)
    buf.seek(0)
    st.download_button("📥 Download trained model (.pkl)", buf, "model.pkl", "application/octet-stream")

# ── TAB 5: Deploy ─────────────────────────────────────────────────────────
with tab5:
    if not st.session_state.get("mlops_trained", False):
        st.info("Train a model first.")
        st.stop()

    st.markdown('<div class="pipe-stage stage-active"><h4>🚀 Stage 5 — Deployment: API + Docker + CI/CD</h4></div>',
                unsafe_allow_html=True)

    dep_tab1, dep_tab2, dep_tab3 = st.tabs(["🌐 API Simulator", "🐳 Dockerfile", "⚡ GitHub Actions CI/CD"])

    with dep_tab1:
        st.markdown("**Simulate a live /predict API call**")
        inp = {}
        cols_d = st.columns(3)
        for i, feat in enumerate(feat_cols[:9]):
            with cols_d[i % 3]:
                mn = float(df[feat].min())
                mx = float(df[feat].max())
                inp[feat] = st.number_input(feat, min_value=mn, max_value=mx,
                                            value=float(df[feat].median()), key=f"api_{feat}")
        if st.button("📡 Call /predict", use_container_width=True):
            X_in = np.array([[inp.get(f, 0) for f in feat_cols]])
            sc_tmp = StandardScaler().fit(df[feat_cols].fillna(0))
            X_sc = sc_tmp.transform(X_in)
            prob = st.session_state.mlops_model.predict_proba(X_sc)[0]
            pred = int(st.session_state.mlops_model.predict(X_sc)[0])
            response = {
                "status": "success",
                "model": st.session_state.mlops_algo,
                "prediction": pred,
                "probability": {
                    "class_0": round(float(prob[0]), 4),
                    "class_1": round(float(prob[1]), 4),
                },
                "input_features": inp,
                "response_time_ms": round(np.random.uniform(12, 45), 1),
            }
            st.markdown(f'<div class="api-box">{json.dumps(response, indent=2)}</div>', unsafe_allow_html=True)
            color = "#EF4444" if pred == 1 else "#10B981"
            risk = "HIGH RISK 🔴" if pred == 1 else "LOW RISK 🟢"
            st.markdown(f"""<div style="margin-top:1rem;padding:1rem;background:{color}15;
            border:2px solid {color};border-radius:10px;text-align:center;">
            <b style="color:{color};font-size:1.1rem">{risk}</b><br>
            <span style="color:#374151;font-size:0.85rem">Confidence: {max(prob)*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)

    with dep_tab2:
        st.markdown("**Production-ready Dockerfile — copy and use directly**")
        dockerfile = """FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]"""
        st.markdown(f'<div class="code-box">{dockerfile}</div>', unsafe_allow_html=True)
        st.download_button("📥 Download Dockerfile", dockerfile, "Dockerfile", "text/plain")

        fastapi_code = '''from fastapi import FastAPI
from pydantic import BaseModel
import pickle, numpy as np
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="AI Nexus MLOps API", version="1.0.0")
model = pickle.load(open("model.pkl", "rb"))

class Features(BaseModel):
    age: float
    income: float
    credit_score: float
    loan_amount: float
    employment_years: float

@app.get("/")
def root():
    return {"status": "AI Nexus MLOps API is live", "docs": "/docs"}

@app.post("/predict")
def predict(features: Features):
    X = np.array([[features.age, features.income, features.credit_score,
                   features.loan_amount, features.employment_years]])
    prob = model.predict_proba(X)[0]
    pred = int(model.predict(X)[0])
    return {"prediction": pred, "probability": {"class_0": round(float(prob[0]),4),
            "class_1": round(float(prob[1]),4)}, "risk": "HIGH" if pred==1 else "LOW"}

@app.get("/health")
def health(): return {"status": "healthy"}'''
        st.markdown("**FastAPI main.py:**")
        st.markdown(f'<div class="code-box">{fastapi_code}</div>', unsafe_allow_html=True)
        st.download_button("📥 Download main.py", fastapi_code, "main.py", "text/plain")

    with dep_tab3:
        st.markdown("**GitHub Actions CI/CD Pipeline — auto-tests on every push**")
        gh_actions = """name: AI Nexus MLOps CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python 3.10
        uses: actions/setup-python@v4
        with: { python-version: "3.10" }
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v --tb=short
      - name: Model quality check
        run: python scripts/validate_model.py

  build-docker:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t ai-nexus-api:${{ github.sha }} .
      - name: Test container
        run: |
          docker run -d -p 8000:8000 ai-nexus-api:${{ github.sha }}
          sleep 5
          curl -f http://localhost:8000/health

  deploy:
    needs: build-docker
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: echo "Deploy step — connect to your cloud provider here"
      - name: Notify team
        run: echo "Deployment complete — model version ${{ github.sha }}"
"""
        st.markdown(f'<div class="code-box">{gh_actions}</div>', unsafe_allow_html=True)
        st.download_button("📥 Download CI/CD workflow", gh_actions, "mlops-pipeline.yml", "text/plain")

        st.markdown("**Run locally with Docker:**")
        cmds = """# Build image
docker build -t ai-nexus-api .

# Run container
docker run -d -p 8000:8000 ai-nexus-api

# Test the API
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"age":35,"income":60000,"credit_score":700,"loan_amount":15000,"employment_years":5}'

# View interactive docs
open http://localhost:8000/docs"""
        st.markdown(f'<div class="code-box">{cmds}</div>', unsafe_allow_html=True)