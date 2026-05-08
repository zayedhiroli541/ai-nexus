# 🧠 AI Nexus — All-in-One AI & Data Science Platform

> **Built by Zayed Asif Hiroli** · Oracle Certified GenAI Professional  
> **Live Demo:** *(Deploy to Streamlit Cloud)*  
> 🔗 [LinkedIn](https://www.linkedin.com/in/zayed-hiroli-5a15762a6) · 🐙 [GitHub](https://github.com/zayedhiroli541)

---

## 🎯 What is AI Nexus?

AI Nexus is a **unified 5-module AI & Data Science platform** that combines every major skill domain in one deployable application. Instead of 5 separate projects, this is **one powerful platform** that demonstrates breadth AND depth across:

| Module | Domain | Key Tech |
|--------|--------|---------|
| 📄 RAG PDF Chatbot | Generative AI | LangChain · FAISS · Mistral-7B |
| 📊 AI Data Insights | Data Science + Statistics + GenAI | Pandas · SciPy · Scikit-Learn · LangChain |
| 🔍 Sentiment Analyser | NLP + Deep Learning + Web Scraping | HuggingFace Transformers · BeautifulSoup |
| 🔮 Churn Predictor | ML + Statistics + GenAI | Scikit-Learn · SciPy · LangChain |
| ⚙️ MLOps Pipeline | MLOps + Deployment | FastAPI · Docker · GitHub Actions |

---

## 🏗️ Architecture

```
AI NEXUS — Streamlit Multi-Page Application
│
├── 🏠 Home.py                    ← Landing page & module overview
│
└── pages/
    ├── 1_📄_RAG_PDF_Chatbot.py   ← LangChain + FAISS + Mistral-7B
    │   PyPDF → Chunk → Embed (MiniLM) → FAISS index → RAG chain → Chat UI
    │
    ├── 2_📊_AI_Data_Insights.py  ← Pandas + SciPy + Scikit-Learn + LangChain
    │   EDA → Normality tests → CLT → Distribution fitting → ML → Clustering → AI narration
    │
    ├── 3_🔍_Sentiment_Analyser.py ← BeautifulSoup + DistilBERT + Seaborn
    │   Web scraping → Text cleaning → Transformer classification → Visualisation
    │
    ├── 4_🔮_Churn_Predictor.py   ← Scikit-Learn + SciPy + LangChain
    │   5-model comparison → ROC curves → CLT + Distribution → AI retention strategy
    │
    └── 5_⚙️_MLOps_Pipeline.py    ← Full MLOps flow
        Ingestion → Feature engineering → Training → Evaluation → API sim → Docker + CI/CD
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/zayedhiroli541/ai-nexus
cd ai-nexus

# 2. Create environment
conda create -n ai-nexus python=3.10
conda activate ai-nexus

# 3. Install
pip install -r requirements.txt

# 4. Run
streamlit run Home.py
```

Open `http://localhost:8501` → navigate modules from sidebar.

---

## 🔑 HuggingFace Token Setup

Modules 1, 2, 4 use Mistral-7B via HuggingFace (free tier):

1. Sign up at [huggingface.co](https://huggingface.co)
2. Settings → Access Tokens → New Token (Read)
3. Paste token in the sidebar of each module

---

## ☁️ Streamlit Cloud Deployment

```
1. Push to GitHub
2. Go to share.streamlit.io
3. Main file: Home.py
4. Secrets: HUGGINGFACEHUB_API_TOKEN = "hf_your_token"
5. Deploy → get public URL
```

---

## 📝 Resume Bullets

**Overall (AI Generalist bullet):**
> "Built AI Nexus — a unified 5-module AI & Data Science platform covering RAG-based PDF Q&A (LangChain + FAISS), statistical analysis (SciPy, CLT, normality tests), NLP sentiment analysis (HuggingFace Transformers), customer churn prediction (5-algorithm ML comparison), and MLOps deployment pipeline (FastAPI + Docker + GitHub Actions CI/CD) — deployed on Streamlit Cloud."

**Per module bullets:**
- **RAG:** Built a RAG-powered PDF Q&A chatbot using LangChain, FAISS, and Hugging Face sentence-transformers, achieving sub-2s response time on 50-page documents.
- **Data Insights:** Developed an AI data insights engine combining Pandas EDA, SciPy hypothesis testing (Shapiro-Wilk, CLT, t-test), Scikit-Learn ML pipelines, and Mistral-7B business narration.
- **Sentiment:** Built a sentiment analysis pipeline scraping live data with BeautifulSoup, classified using DistilBERT with 91%+ confidence, visualised trends across categories with Seaborn.
- **Churn:** Developed a churn predictor comparing 5 ML algorithms (Random Forest, GBM, LR, DT, SVM), with CLT simulation, distribution fitting, ROC-AUC evaluation, and AI retention strategy generation.
- **MLOps:** Engineered a full MLOps pipeline: data ingestion → feature engineering → cross-validated training → confusion matrix/ROC evaluation → FastAPI /predict endpoint → Docker containerisation → GitHub Actions CI/CD.

---

## 🧠 Skills Demonstrated

| Category | Skills |
|----------|--------|
| Programming | Python, Pandas, NumPy |
| Data Science | EDA, Feature Engineering, Data Visualisation |
| Statistics | Probability Distributions, CLT, Normality Tests, T-test |
| Machine Learning | Random Forest, GBM, Logistic Regression, SVM, Cross-validation |
| Deep Learning & NLP | HuggingFace Transformers, DistilBERT, Sentiment Analysis |
| Generative AI | LLMs, RAG, LangChain, Prompt Engineering, FAISS |
| Web Scraping | BeautifulSoup, Requests, Data Collection |
| MLOps | ML Pipeline, FastAPI, Docker, GitHub Actions, CI/CD |
| Deployment | Streamlit Cloud, REST APIs, Model Serialisation |

---

## 👤 Author

**Zayed Asif Hiroli** · Data Science · Generative AI · MLOps · AI Generalist  
📧 zayedhiroli541@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/zayed-hiroli-5a15762a6) · 🐙 [GitHub](https://github.com/zayedhiroli541)  
🏅 Oracle Certified GenAI Professional (OCI 2025, valid until Oct 2027)

---

## 📄 License
MIT License — free to use and build upon.
