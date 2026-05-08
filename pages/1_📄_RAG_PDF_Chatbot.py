"""Module 1 — RAG PDF Q&A Chatbot (Groq – free & fast)"""
import streamlit as st
import os, time, tempfile, requests
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG PDF Chatbot | AI Nexus", page_icon="📄", layout="wide")

st.markdown("""
<style>
.mod-header { background: linear-gradient(135deg,#4338CA,#6366F1);
  padding:1.5rem 2rem; border-radius:12px; color:white; margin-bottom:1.5rem; }
.mod-header h2 { margin:0; font-size:1.5rem; font-weight:700; }
.mod-header p  { margin:0.3rem 0 0; opacity:0.8; font-size:0.88rem; }
.source-box { background:#F0F4FF; border-left:4px solid #6366F1;
  padding:0.7rem 1rem; border-radius:0 8px 8px 0; font-size:0.82rem;
  color:#1E293B; margin-top:5px; }
.metric-row { display:flex; gap:10px; margin-bottom:1rem; }
.metric-chip { background:#EEF2FF; border-radius:8px; padding:0.6rem 1rem;
  text-align:center; flex:1; }
.metric-chip .val { font-size:1.4rem; font-weight:700; color:#4338CA; }
.metric-chip .lbl { font-size:0.7rem; color:#64748B; }
.groq-badge { background:#F0FDF4; border:1px solid #86EFAC; border-radius:8px;
  padding:0.65rem 1rem; font-size:0.82rem; color:#14532D; margin-bottom:1rem; }
.step-box { background:#F8FAFF; border:1px solid #E2E8F5; border-radius:10px;
  padding:0.9rem 1.1rem; margin-bottom:0.6rem; font-size:0.85rem; color:#1E293B; }
.step-box b { color:#4338CA; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mod-header">
  <h2>📄 Module 1 — RAG PDF Q&A Chatbot</h2>
  <p>Upload any PDF → ask natural language questions → get answers with exact page citations</p>
  <p><strong>Stack:</strong> LangChain · FAISS · HuggingFace Embeddings (MiniLM-L6) · Groq Llama‑3 · Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────
for k, v in {
    "rag_vs": None,
    "rag_groq_key": None,
    "rag_messages": [],
    "rag_processed": False,
    "rag_pages": 0,
    "rag_chunks": 0,
    "rag_queries": 0,
    "rag_pdf_name": "",
    "rag_model": "llama-3.1-8b-instant",   # will be updated dynamically
    "rag_pending": None,                    # used for suggestion buttons
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Dynamic model fetching ─────────────────────────────────────────────────
FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

@st.cache_data(ttl=86400)  # cache for 24 hours
def fetch_available_groq_models(api_key: str):
    """Return list of active, non-preview, chat models from Groq. Falls back to hardcoded list on error."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            available = []
            for m in data.get("data", []):
                if not m.get("active", False):
                    continue
                if "chat" not in m.get("type", ""):
                    continue
                if "preview" in m.get("id", "").lower():
                    continue
                available.append(m["id"])
            if available:
                return available
    except Exception:
        pass
    return FALLBACK_MODELS


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        path = tmp.name
    pages  = PyPDFLoader(path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(pages)
    vs     = FAISS.from_documents(chunks, get_embeddings())
    os.unlink(path)
    return vs, len(pages), len(chunks)


def ask_groq(api_key: str, context: str, question: str,
             history: list, model_list: list) -> str:
    """Try each model in model_list until one succeeds."""
    client = Groq(api_key=api_key)
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions based ONLY on the "
            "provided document context. Be specific and cite relevant details from the context."
        )
    }
    messages = [system_msg]
    for m in history[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({
        "role": "user",
        "content": (
            f"Context from the document:\n\n{context}\n\n---\n\n"
            f"Question: {question}\n\nAnswer based on the context above."
        )
    })

    last_error = None
    for model in model_list:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            if "model_decommissioned" in str(e).lower() or "404" in str(e) or "400" in str(e):
                continue  # try next model
            else:
                raise e
    raise last_error if last_error else RuntimeError("No model worked")


# ── Answer generation helper ─────────────────────────────────────────────
def generate_answer_for_question(user_q: str):
    """Run retrieval + Groq generation for the given question and update chat history."""
    t0 = time.time()
    # Retrieve top-4 chunks
    sources = st.session_state.rag_vs.similarity_search(user_q, k=4)
    context = "\n\n---\n\n".join(
        [f"[Page {doc.metadata.get('page','?')}]\n{doc.page_content}" for doc in sources]
    )

    # Build fallback order: selected model first, then remaining
    history_for_llm = st.session_state.rag_messages[:-1]  # exclude the current user message just added
    priority_models = [st.session_state.rag_model]
    for m in FALLBACK_MODELS:
        if m not in priority_models:
            priority_models.append(m)

    answer = ask_groq(
        api_key=st.session_state.rag_groq_key,
        context=context,
        question=user_q,
        history=history_for_llm,
        model_list=priority_models,
    )

    elapsed = round(time.time() - t0, 2)

    # Store assistant message
    st.session_state.rag_queries += 1
    st.session_state.rag_messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "time": elapsed
    })
    return answer, sources, elapsed


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Module 1 Setup")

    # Groq key (auto‑fills from secrets.toml if available)
    groq_key = st.text_input(
        "🔑 Groq API Key",
        value=st.secrets.get("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com"
    )
    if groq_key:
        st.session_state.rag_groq_key = groq_key
        st.success("✅ Groq key set")

        # ── Dynamic model selector ──────────────────────────────────────────
        available_models = fetch_available_groq_models(groq_key)
        # Keep a valid model in session state
        if st.session_state.rag_model not in available_models:
            st.session_state.rag_model = available_models[0]

        st.markdown("**🤖 Model**")
        model_choice = st.selectbox(
            "Choose model",
            options=available_models,
            index=available_models.index(st.session_state.rag_model)
                  if st.session_state.rag_model in available_models else 0,
            help=(
                "Models are fetched live from Groq. Only active, free chat models are shown.\n"
                "If your selected model fails, the app will automatically try other available models."
            )
        )
        st.session_state.rag_model = model_choice

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

    st.markdown("---")
    st.markdown("**📄 Upload PDF**")
    uploaded = st.file_uploader("Any PDF up to 50MB", type=["pdf"])

    if uploaded and groq_key:
        if st.button("🚀 Process PDF", use_container_width=True):
            with st.spinner("Loading embeddings + indexing PDF..."):
                try:
                    vs, pages, chunks = process_pdf(uploaded)
                    st.session_state.rag_vs        = vs
                    st.session_state.rag_processed = True
                    st.session_state.rag_pages     = pages
                    st.session_state.rag_chunks    = chunks
                    st.session_state.rag_pdf_name  = uploaded.name
                    st.session_state.rag_messages  = []
                    st.session_state.rag_queries   = 0
                    st.success(f"✅ {pages} pages · {chunks} chunks")
                except Exception as e:
                    st.error(f"❌ PDF Error: {e}")
    elif uploaded and not groq_key:
        st.warning("⚠️ Add your Groq API key first")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.rag_messages = []
        st.session_state.rag_queries  = 0
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════
status = "✅" if st.session_state.rag_processed else "⏳"
st.markdown(f"""
<div class="metric-row">
  <div class="metric-chip"><div class="val">{st.session_state.rag_pages}</div><div class="lbl">Pages loaded</div></div>
  <div class="metric-chip"><div class="val">{st.session_state.rag_chunks}</div><div class="lbl">Chunks embedded</div></div>
  <div class="metric-chip"><div class="val">{st.session_state.rag_queries}</div><div class="lbl">Queries answered</div></div>
  <div class="metric-chip"><div class="val">{status}</div><div class="lbl">Status</div></div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.rag_processed:
    st.markdown('<div class="groq-badge">✅ <b>Using Groq API</b> — models are dynamically fetched & always up‑to‑date</div>', unsafe_allow_html=True)
    st.markdown("### How this RAG pipeline works")
    steps = [
        ("📥 Load",    "PDF parsed page by page"),
        ("✂️ Chunk",   "Text split into 1000‑char overlapping chunks"),
        ("🔢 Embed",   "Each chunk → vector with MiniLM‑L6‑v2 (local, free)"),
        ("🗄️ Index",   "All vectors stored in FAISS for instant search"),
        ("❓ Query",   "Your question vectorised → top‑4 chunks retrieved"),
        ("🤖 Generate","Groq model generates answer from chunks (automatic fallback if needed)"),
        ("📚 Cite",    "Source chunks shown with page numbers"),
    ]
    for icon_label, desc in steps:
        st.markdown(f'<div class="step-box"><b>{icon_label}</b> — {desc}</div>', unsafe_allow_html=True)
else:
    model_label = st.session_state.rag_model
    st.markdown(f'<div class="groq-badge">✅ PDF loaded: <b>{st.session_state.rag_pdf_name}</b> | Model: <b>{model_label}</b> via Groq</div>', unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.rag_messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} chunks) · ⏱ {msg.get('time','?')}s"):
                    for i, doc in enumerate(msg["sources"], 1):
                        pg = doc.metadata.get("page", "?")
                        st.markdown(f"<div class='source-box'><b>Chunk {i} — Page {pg}</b><br>{doc.page_content[:400]}...</div>", unsafe_allow_html=True)

    # ── Suggestion buttons ─────────────────────────────────────────────────
    if not st.session_state.rag_messages:
        st.markdown("💡 **Try asking:**")
        suggestions = [
            "What is the main topic?",
            "Summarise key findings",
            "List all recommendations",
            "What are the conclusions?",
        ]
        c1, c2 = st.columns(2)
        for i, q in enumerate(suggestions):
            with (c1 if i % 2 == 0 else c2):
                if st.button(q, key=f"sug_{i}"):
                    st.session_state.rag_messages.append({"role": "user", "content": q})
                    st.session_state.rag_pending = q   # set flag to trigger generation
                    st.rerun()

    # ── Handle pending suggestion (if any) ─────────────────────────────────
    if st.session_state.get("rag_pending"):
        user_q = st.session_state.rag_pending
        st.session_state.rag_pending = None   # clear flag

        # Display the user message (already in history, but we still show it)
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_q)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching document + generating answer..."):
                try:
                    answer, sources, elapsed = generate_answer_for_question(user_q)
                    st.markdown(answer)

                    with st.expander(f"📚 Sources ({len(sources)} chunks) · ⏱ {elapsed}s"):
                        for i, doc in enumerate(sources, 1):
                            pg = doc.metadata.get("page", "?")
                            st.markdown(f"<div class='source-box'><b>Chunk {i} — Page {pg}</b><br>{doc.page_content[:400]}...</div>", unsafe_allow_html=True)
                except Exception as e:
                    err = str(e)
                    st.error(f"❌ {err}")
                    if "401" in err or "auth" in err.lower():
                        st.warning("💡 Invalid API key. Get a free one at console.groq.com")
                    elif "429" in err:
                        st.warning("💡 Rate limit hit. Free tier: 30 req/min. Wait a moment and try again.")
                    else:
                        st.warning("💡 All available models failed. Please check your API key or try again later.")
        # Stop further processing so we don't show the chat input widget again before the answer
        st.stop()

    # ── Chat input (works when no pending suggestion) ──────────────────────
    if user_q := st.chat_input("Ask anything about your PDF..."):
        st.session_state.rag_messages.append({"role": "user", "content": user_q})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_q)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching document + generating answer..."):
                try:
                    answer, sources, elapsed = generate_answer_for_question(user_q)
                    st.markdown(answer)

                    with st.expander(f"📚 Sources ({len(sources)} chunks) · ⏱ {elapsed}s"):
                        for i, doc in enumerate(sources, 1):
                            pg = doc.metadata.get("page", "?")
                            st.markdown(f"<div class='source-box'><b>Chunk {i} — Page {pg}</b><br>{doc.page_content[:400]}...</div>", unsafe_allow_html=True)
                except Exception as e:
                    err = str(e)
                    st.error(f"❌ {err}")
                    if "401" in err or "auth" in err.lower():
                        st.warning("💡 Invalid API key. Get a free one at console.groq.com")
                    elif "429" in err:
                        st.warning("💡 Rate limit hit. Free tier: 30 req/min. Wait a moment and try again.")
                    else:
                        st.warning("💡 All available models failed. Please check your API key or try again later.")