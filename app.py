import os
import pickle
import numpy as np
import streamlit as st
from datetime import datetime
from deep_translator import GoogleTranslator
from langdetect import detect
from sentence_transformers import SentenceTransformer

# Conditional import for faiss
try:
    import faiss
except ImportError:
    faiss = None
    st.warning("FAISS is not installed; knowledge base search will be disabled.")

# Conditional import for torch
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    st.warning("Torch or Transformers are not installed; fallback FLAN generation disabled.")

# ===== GLOBAL STATE =====
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== CONFIG =====
FAISS_DOCS = "fast_guvi_docs (1).pkl"
FAISS_IDX = "fast_guvi_index (1).index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINE_TUNED_MODEL = "Satya28Kanth/guvi_flan_finetuned"
CACHED_MODEL_DIR = "./cached_model"

INDIAN_LANGUAGES = {
    "en": "English", "hi": "हिंदी", "ta": "தமிழ்", "te": "తెలుగు",
    "kn": "ಕನ್ನಡ", "ml": "മലയാളം", "gu": "ગુજરાતી", "pa": "ਪੰਜਾਬੀ", "mr": "मराठी"
}

# ===== LOAD KB =====
@st.cache_resource
def load_kb():
    if faiss is None:
        return None, None, None
    with open(FAISS_DOCS, "rb") as f:
        docs = pickle.load(f)
    index = faiss.read_index(FAISS_IDX)
    embedder = SentenceTransformer(EMBED_MODEL)
    return docs, index, embedder

docs, index, embedder = load_kb()

# ===== LOAD FLAN MODEL (LOCAL CACHE FIRST) =====
@st.cache_resource
def load_fallback_model():
    if torch is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        return None, None
    try:
        if os.path.exists(CACHED_MODEL_DIR):
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(CACHED_MODEL_DIR, "tokenizer"))
            model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(CACHED_MODEL_DIR, "model"))
            st.sidebar.success("✅ FLAN loaded from local cache")
        else:
            hf_token = os.getenv("HF_TOKEN")
            tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL, use_auth_token=hf_token)
            model = AutoModelForSeq2SeqLM.from_pretrained(FINE_TUNED_MODEL, use_auth_token=hf_token)
            st.sidebar.info("⬇️ Downloaded FLAN model from Hugging Face")
        return tokenizer, model
    except Exception as e:
        st.sidebar.warning(f"⚠️ FLAN fallback not loaded: {e}")
        return None, None

FALLBACK_TOKENIZER, FALLBACK_MODEL = load_fallback_model()

# ===== LANG DETECT & TRANSLATE =====
def detect_lang(text):
    try:
        lang = detect(text)
        return lang if lang in INDIAN_LANGUAGES else "en"
    except:
        return "en"

def translate(text, src, tgt):
    if src == tgt:
        return text
    try:
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except:
        return text

# ===== FAISS KB SEARCH =====
def search_kb(query_en, top_k=1):
    if faiss is None or docs is None or index is None or embedder is None:
        return []
    qvec = embedder.encode([query_en], normalize_embeddings=True).astype('float32')
    scores, idxs = index.search(qvec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs):
        if idx != -1:
            results.append((docs[int(idx.item())], score))
    return results

# ===== INTENT CLASSIFICATION =====
def detect_intent(query_en):
    career_keywords = ["career", "job", "path", "guidance", "goal"]
    course_keywords = ["course", "learn", "skill", "technology"]
    if any(k in query_en.lower() for k in career_keywords):
        return "career_guidance"
    elif any(k in query_en.lower() for k in course_keywords):
        return "course_recommendation"
    return "general"

# ===== FLAN FALLBACK GENERATION =====
def generate_with_flan(query_en, kb_results):
    if FALLBACK_MODEL is None or FALLBACK_TOKENIZER is None:
        return "FLAN model is not available. Please contact support."
    context = "\n".join([r[0]["text"] for r in kb_results[:3]]) if kb_results else ""
    prompt = (
        "You are an expert assistant for GUVI courses and career guidance. "
        "Answer the user's question clearly and professionally using the context below. "
        "If you don't have the information, say you don't know and suggest contacting GUVI support at +91-93635 21396.\n\n"
        f"User question:\n{query_en}\n\nContext:\n{context}\n\nAnswer:"
    )
    inputs = FALLBACK_TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = FALLBACK_MODEL.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        eos_token_id=FALLBACK_TOKENIZER.eos_token_id,
        pad_token_id=FALLBACK_TOKENIZER.pad_token_id
    )
    answer = FALLBACK_TOKENIZER.decode(outputs[0], skip_special_tokens=True).strip()
    if len(answer) < 20:
        answer = "I'm sorry, I currently don’t have detailed information. Please reach out to GUVI support at +91-93635 21396."
    return answer

# ===== UI =====
st.set_page_config(page_title="GUVI Multilingual Assistant", page_icon="🇮🇳", layout="wide")

st.markdown("""
<div style='background: linear-gradient(90deg,#24b47e,#b0f2c2);
padding:1.5rem; border-radius:12px; text-align:center;
font-size:30px; font-weight:bold; color:#222; box-shadow:0 4px 18px #24b47e20; letter-spacing:1px;'>
<span style='color:#138808'>GUVI</span> Multilingual Assistant
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
selected_lang = st.sidebar.selectbox("🌐 Language", ["auto"] + list(INDIAN_LANGUAGES.keys()),
                                    format_func=lambda x: "Auto Detect" if x == "auto" else INDIAN_LANGUAGES[x])
show_context = st.sidebar.checkbox("Show Source Context", value=False)
enable_debug = st.sidebar.checkbox("Enable Debug Panel", value=False)
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    role = '🧑‍💻 User' if msg['role'] == 'user' else '🤖 Assistant'
    st.markdown(f"**{role} [{msg['time']}]:** {msg['content']}")
    if msg.get("context") and show_context:
        st.markdown(f"<div style='font-size:12px; background:#f9f9f9; padding:6px; border-radius:5px;'>"
                    f"<b>Source ({msg.get('source','')}):</b><br>{msg['context']}</div>", unsafe_allow_html=True)

# Input Form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("आपका प्रश्न | Your Question", "")
    submitted = st.form_submit_button("🚀 Ask")

if submitted and user_input.strip():
    user_lang = detect_lang(user_input) if selected_lang == "auto" else selected_lang
    query_en = translate(user_input, user_lang, "en")
    intent = detect_intent(query_en)

    # Knowledge Base Search
    kb_results = search_kb(query_en, top_k=1)
    best_doc, score = (kb_results[0] if kb_results else (None, 0))

    # Debug Info
    debug_data = {
        "Detected Lang": user_lang,
        "Translated Query": query_en,
        "Intent": intent,
        "KB Hit": bool(best_doc),
        "KB Confidence": round(float(score.item()), 3) if score else 0
    }
    if enable_debug:
        st.sidebar.markdown("### 🛠 Debug Info")
        for k, v in debug_data.items():
            st.sidebar.write(f"**{k}:** {v}")

    # Decide KB vs Fallback
    if best_doc and score >= 0.75:
        answer_en = best_doc["answer"]
        src_context, source_type = f"Q: {best_doc['question']}\nA: {best_doc['answer']}", "KB"
    else:
        answer_en = generate_with_flan(query_en, kb_results)
        src_context, source_type = "", "FLAN" if FALLBACK_MODEL else "NONE"

    final_answer = translate(answer_en, "en", user_lang)

    # Store history
    st.session_state.messages.append({"role": "user", "content": user_input, "time": datetime.now().strftime("%H:%M:%S")})
    st.session_state.messages.append({"role": "assistant", "content": final_answer, "time": datetime.now().strftime("%H:%M:%S"),
                                     "context": src_context, "source": source_type})

    # Display
    st.markdown("---")
    st.subheader(f"📢 Answer ({'Knowledge Base' if source_type=='KB' else 'AI Fallback'})")
    st.write(final_answer)
    if show_context and src_context:
        st.code(src_context)
    st.markdown("---")
