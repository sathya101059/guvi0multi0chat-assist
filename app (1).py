# =================== SET WRITABLE CACHE FIRST ===================
import os
CACHE_DIR = "/tmp/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

# ===============================
# 2️⃣ IMPORT STREAMLIT FIRST
# ===============================
import streamlit as st

# Must be FIRST Streamlit command after import
st.set_page_config(page_title="GUVI Multilingual Assistant", page_icon="🇮🇳", layout="wide")

# =================== IMPORTS ===================
import pickle
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datetime import datetime

# =================== GLOBAL STATE ===================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =================== CONFIG ===================
FAISS_DOCS = "fast_guvi_docs (1).pkl"   # Uploaded in HF Space
FAISS_IDX = "fast_guvi_index (1).index"
KB_EMB_FILE = "kb_embeddings.npy"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FLAN_MODEL_NAME = "google/flan-t5-small"
CONFIDENCE_THRESHOLD = 0.7

INDIAN_LANGUAGES = {
    "en": "English", "hi": "हिंदी", "ta": "தமிழ்", "te": "తెలుగు",
    "kn": "ಕನ್ನಡ", "ml": "മലയാളം", "gu": "ગુજરાતી", "pa": "ਪੰਜਾਬੀ", "mr": "मराठी"
}

# =================== LOAD KB ===================
@st.cache_resource
def load_kb():
    with open(FAISS_DOCS, "rb") as f:
        docs = pickle.load(f)
    index = faiss.read_index(FAISS_IDX)
    embeddings = np.load(KB_EMB_FILE)
    embedder = SentenceTransformer(EMBED_MODEL, cache_folder=CACHE_DIR)
    return docs, index, embeddings, embedder

docs, faiss_index, kb_embeddings, embedder = load_kb()

# =================== LOAD FLAN MODEL ===================
@st.cache_resource
def load_flan():
    try:
        tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL_NAME, cache_dir=CACHE_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL_NAME, cache_dir=CACHE_DIR)
        st.sidebar.success("✅ FLAN model loaded")
        return tokenizer, model
    except Exception as e:
        st.sidebar.warning(f"⚠️ FLAN model could not be loaded:\n{e}")
        return None, None

flan_tokenizer, flan_model = load_flan()

# =================== HELPER FUNCTIONS ===================
def detect_lang(text):
    try:
        lang = detect(text)
        return lang if lang in INDIAN_LANGUAGES else "en"
    except:
        return "en"

def translate(text, target_lang="en"):
    if not text.strip():
        return ""
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except:
        return text

def search_kb(query, top_k=3):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = faiss_index.search(np.array(q_emb), top_k)
    results = [(docs[int(idx)], float(scores[0][i])) for i, idx in enumerate(idxs[0]) if idx != -1]
    return results

def generate_with_flan(prompt):
    if flan_model is None or flan_tokenizer is None:
        return "FLAN model not available"
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = flan_model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        eos_token_id=flan_tokenizer.eos_token_id,
        pad_token_id=flan_tokenizer.pad_token_id
    )
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# =================== STREAMLIT UI ===================

st.markdown("""
<div style='background: linear-gradient(90deg,#24b47e,#b0f2c2);
padding:1.5rem; border-radius:12px; text-align:center;
font-size:30px; font-weight:bold; color:#222; box-shadow:0 4px 18px #24b47e20; letter-spacing:1px;'>
<span style='color:#138808'>GUVI</span> Multilingual Assistant
</div>
""", unsafe_allow_html=True)

# Sidebar
selected_lang = st.sidebar.selectbox("🌐 Language", ["auto"] + list(INDIAN_LANGUAGES.keys()),
                                    format_func=lambda x: "Auto Detect" if x=="auto" else INDIAN_LANGUAGES[x])
show_context = st.sidebar.checkbox("Show Source Context", value=False)
enable_debug = st.sidebar.checkbox("Enable Debug Panel", value=False)
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask your question:", "")
    submitted = st.form_submit_button("🚀 Ask")

if submitted and user_input.strip():
    user_lang = detect_lang(user_input) if selected_lang=="auto" else selected_lang
    query_en = translate(user_input, target_lang="en")

    kb_results = search_kb(query_en, top_k=1)
    best_doc, score = (kb_results[0] if kb_results else (None, 0))

    if enable_debug:
        st.sidebar.markdown("### 🛠 Debug Info")
        st.sidebar.write({
            "Detected Lang": user_lang,
            "Translated Query": query_en,
            "KB Hit": bool(best_doc),
            "KB Score": round(score, 3)
        })

    if best_doc and score >= CONFIDENCE_THRESHOLD:
        answer_en = best_doc.get("answer", "No answer found.")
        final_answer = translate(answer_en, target_lang=user_lang)
        src_context = f"Q: {best_doc['question']}\nA: {best_doc['answer']}"
        source_type = "KB"
    else:
        flan_prompt = f"Answer the question: {query_en}"
        answer_en = generate_with_flan(flan_prompt)
        final_answer = translate(answer_en, target_lang=user_lang)
        src_context = ""
        source_type = "FLAN" if flan_model else "NONE"

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "time": datetime.now().strftime("%H:%M:%S")
    })
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "time": datetime.now().strftime("%H:%M:%S"),
        "context": src_context,
        "source": source_type
    })

    # Display chat
    for msg in st.session_state.messages[-4:]:
        role = '🧑‍💻 User' if msg['role']=='user' else '🤖 Assistant'
        st.markdown(f"**{role} [{msg['time']}]:** {msg['content']}")
        if show_context and msg.get("context"):
            st.code(msg.get("context"))

