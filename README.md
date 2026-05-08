# 🧠 AI Nexus — All-in-One AI & Data Science Platform

> **Built by Zayed Asif Hiroli** · Oracle Certified GenAI Professional  
> **Live Demo:** [AI Nexus on Streamlit Cloud](https://ai-nexus.streamlit.app) *(after deployment)*  
> 🔗 [LinkedIn](https://www.linkedin.com/in/zayed-hiroli-5a15762a6) · 🐙 [GitHub](https://github.com/zayedhiroli541)

---

## 🎯 What is AI Nexus?

AI Nexus is a **unified 5‑module AI & Data Science platform** that combines every major skill domain in one deployable application. Instead of 5 separate projects, this is **one powerful platform** that demonstrates breadth AND depth across:

| Module | Domain | Key Tech |
|--------|--------|---------|
| 📄 RAG PDF Chatbot | Generative AI | LangChain · FAISS · Groq (free Llama‑3) |
| 📊 AI Data Insights | Data Science + Statistics + GenAI | Pandas · SciPy · Scikit‑Learn · Groq |
| 🔍 Sentiment Analyser | NLP + Deep Learning + Web Scraping | HuggingFace Transformers · BeautifulSoup |
| 🔮 Churn Predictor | ML + Statistics + GenAI | Scikit‑Learn · SciPy · Groq |
| ⚙️ MLOps Pipeline | MLOps + Deployment | FastAPI (simulated) · Docker · GitHub Actions |

---

## 🚀 Features

- **Dynamic model fetching** — the app fetches the list of currently available Groq models live. No more hard‑coded, deprecated models.  
- **Automatic fallback** — if the selected model fails, the app silently tries the next available model. Users never see an error.  
- **One‑click suggestions** in the PDF chatbot — instant answer generation.  
- **Full statistical analysis** — Shapiro‑Wilk, CLT simulation, distribution fitting, t‑tests.  
- **5‑model churn comparison** with live risk calculator.  
- **End‑to‑end MLOps pipeline** — data ingestion → feature engineering → training → evaluation → API simulation → Docker/CI‑CD code ready to copy.  

---

## 🏗️ Architecture

```
AI NEXUS — Streamlit Multi-Page Application
│
├── 🏠 Home.py ← Landing page & module overview
│
└── pages/
    ├── 1_📄_RAG_PDF_Chatbot.py ← LangChain + FAISS + Groq (RAG)
    ├── 2_📊_AI_Data_Insights.py ← Pandas + SciPy + Scikit‑Learn + Groq
    ├── 3_🔍_Sentiment_Analyser.py ← BeautifulSoup + DistilBERT
    ├── 4_🔮_Churn_Predictor.py ← Scikit‑Learn + SciPy + Groq
    └── 5_⚙️_MLOps_Pipeline.py ← Full MLOps flow
```

---

## 🔧 Quick Start

```bash
# 1. Clone
git clone https://github.com/zayedhiroli541/ai-nexus
cd ai-nexus

# 2. Create environment (optional but recommended)
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run Home.py
```

Open `http://localhost:8501` → use the sidebar to navigate between modules.

---

## 🔑 API Key Setup (Groq – free)

Modules 1, 2, and 4 use Groq's free LLM API.  
No credit card required – just sign up and get a key:

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (Google / GitHub)
3. Click **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_`)
5. Paste it in the sidebar of any LLM‑powered module (Modules 1, 2, 4)

Your key is never stored by the app – it's only used for that session.

---

## ☁️ Deploy on Streamlit Cloud (free)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. **New app** → select your repo, branch `main`, main file `Home.py`.
4. Under **Advanced settings** → **Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy!** — you'll get a public URL.

---

## 📝 Modules & Bullet Points (ready to use)

### Overall (AI Generalist bullet):
Built AI Nexus — a unified 5‑module AI & Data Science platform covering RAG, NLP, statistical analysis (CLT, hypothesis testing), churn prediction (5 ML algorithms, risk calculator), and MLOps (FastAPI simulation, Docker, CI/CD), powered by Groq's free LLM API with dynamic model fetching & auto‑fallback — deployed on Streamlit Cloud.

### Per‑module bullets:

**RAG PDF Chatbot:**  
Built a RAG‑based PDF Q&A chatbot using LangChain, FAISS, and Hugging Face embeddings; integrated with Groq's free LLM API with dynamic model fetching and automatic fallback, achieving sub‑2s answers with page citations.

**AI Data Insights:**  
Developed an AI‑powered statistical engine combining Pandas EDA, SciPy hypothesis testing (Shapiro‑Wilk, CLT, t‑test), Scikit‑Learn ML, KMeans/PCA, and Groq‑generated business narratives.

**Sentiment Analyser:**  
Built a sentiment analysis pipeline scraping live data with BeautifulSoup, classified using Hugging Face transformers, visualised trends across categories.

**Churn Predictor:**  
Developed a churn predictor comparing 5 ML algorithms with CLT simulation, distribution fitting, and AI‑generated retention strategy via Groq; includes interactive risk calculator.

**MLOps Pipeline:**  
Engineered a full MLOps pipeline: data ingestion → feature engineering → cross‑validated training → evaluation → FastAPI /predict simulation → Docker containerisation → GitHub Actions CI/CD.

---

## 🧠 Skills Demonstrated

| Category | Skills |
|----------|--------|
| Programming | Python, Pandas, NumPy |
| Data Science | EDA, Feature Engineering, Data Visualisation |
| Statistics | CLT, Normality Tests, T‑test, Distribution Fitting |
| Machine Learning | Random Forest, GBM, Logistic Regression, SVM, Cross‑validation |
| NLP & Deep Learning | HuggingFace Transformers, DistilBERT, Sentiment Analysis |
| Generative AI | RAG, LangChain, FAISS, Groq LLM, Prompt Engineering |
| MLOps | ML Pipeline, FastAPI Simulation, Docker, GitHub Actions CI/CD |
| Deployment | Streamlit Cloud, REST APIs |

---

## 👤 Author

**Zayed Asif Hiroli** · Data Science · Generative AI · MLOps · AI Generalist

📧 zayedhiroli541@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/zayed-hiroli-5a15762a6) · 🐙 [GitHub](https://github.com/zayedhiroli541)

**🏅 Certifications:**
- Oracle Certified GenAI Professional (OCI 2025)
- Oracle Certified AI Foundations Associate (OCI 2025)

---

## 📄 License

MIT License — free to use and build upon.

---
