import os
import pickle
import faiss
import numpy as np
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ===== CONFIG =====
FAISS_DOCS = "fast_guvi_docs (1).pkl"  
FAISS_IDX = "fast_guvi_index (1).index"  
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINE_TUNED_MODEL = "Satya28Kanth/guvi_flan_finetuned"

INDIAN_LANGUAGES = {
    "en": "English", "hi": "हिंदी", "ta": "தமிழ்", "te": "తెలుగు",
    "kn": "ಕನ್ನಡ", "ml": "മലയാളം", "gu": "ગુજરાતી", "pa": "ਪੰਜਾਬੀ", "mr": "मराठी"
}

# ===== LOAD KB =====
@st.cache_resource
def load_kb():
    with open(FAISS_DOCS, "rb") as f:
        docs = pickle.load(f)
    index = faiss.read_index(FAISS_IDX)
    embedder = SentenceTransformer(EMBED_MODEL)
    return docs, index, embedder

docs, index, embedder = load_kb()

# ===== LOAD OPTIONAL FLAN MODEL =====
@st.cache_resource
def load_fallback_model():
    try:
        hf_token = os.getenv("HF_TOKEN")
        tok = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL, token=hf_token)
        mod = AutoModelForSeq2SeqLM.from_pretrained(FINE_TUNED_MODEL, token=hf_token)
        st.sidebar.success("✅ FLAN fallback loaded")
        return tok, mod
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
    qvec = embedder.encode([query_en], normalize_embeddings=True).astype('float32')
    scores, idxs = index.search(qvec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx != -1:
            results.append((docs[idx], score))
    return results

# ===== CAREER GUIDANCE LOGIC =====
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
    if not FALLBACK_MODEL:
        return None

    # Prepare context by joining top 3 KB texts if available
    context = "\n".join([r[0]["text"] for r in kb_results[:3]]) if kb_results else ""

    prompt = (
        "You are an expert assistant for GUVI courses and career guidance. "
        "Answer the user's question clearly and professionally using the context below. "
        "If you don't have the precise information, say you don't know and suggest contacting GUVI support at +91-93635 21396.\n\n"
        f"User question:\n{query_en}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
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

    # Replace overly short or unhelpful answers with a polite fallback message
    if len(answer) < 20 or any(phrase in answer.lower() for phrase in ["don't know", "no information", "not available"]):
        answer = (
            "I'm sorry, I currently don't have detailed information on this topic. "
            "Please reach out to GUVI support at +91-93635 21396 for assistance."
        )

    return answer


# ===== UI =====
st.set_page_config(page_title="GUVI", page_icon="🇮🇳", layout="wide")

st.markdown("""
<div style='background: linear-gradient(90deg, #24b47e 0%, #b0f2c2 100%);
    padding: 1.5rem; border-radius: 12px; text-align: center; 
    font-size: 30px; font-weight: bold; color: #222; 
    box-shadow: 0 4px 18px #24b47e20; letter-spacing: 1px;'>
    <span style='color:#138808'>GUVI</span> Multilingual Assistant
</div>
""", unsafe_allow_html=True)


# Sidebar
selected_lang = st.sidebar.selectbox("🌐 Language", ["auto"] + list(INDIAN_LANGUAGES.keys()),
                                     format_func=lambda x: "Auto Detect" if x == "auto" else INDIAN_LANGUAGES[x])
show_context = st.sidebar.checkbox("Show Source Context", value=False)
enable_debug = st.sidebar.checkbox("Enable Debug Panel", value=False)
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = '🧑‍💻 User' if msg['role'] == 'user' else '🤖 Assistant'
    st.markdown(f"**{role} [{msg['time']}]:** {msg['content']}")
    if msg.get("context") and show_context:
        st.markdown(
            f"<div style='font-size:12px; background:#f9f9f9; padding:6px; border-radius:5px;'>"
            f"<b>Source ({msg.get('source','')}):</b><br>{msg['context']}</div>",
            unsafe_allow_html=True
        )

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("आपका प्रश्न | Your Question", "")
    submitted = st.form_submit_button("🚀 Ask")

if submitted and user_input.strip():
    # Detect & translate
    user_lang = detect_lang(user_input) if selected_lang == "auto" else selected_lang
    query_en = translate(user_input, user_lang, "en")
    intent = detect_intent(query_en)

    # Search KB
    kb_results = search_kb(query_en, top_k=1)
    if kb_results:
        best_doc, score = kb_results[0]
        confidence = score
    else:
        best_doc, confidence = None, 0

    # Debug info
    debug_data = {
        "Detected Lang": user_lang,
        "Translated Query": query_en,
        "Intent": intent,
        "KB Hit": bool(best_doc),
        "KB Confidence": round(confidence, 3)
    }

    if enable_debug:
        st.sidebar.markdown("### 🛠 Debug Info")
        for k, v in debug_data.items():
            st.sidebar.write(f"**{k}:** {v}")

    # Decide KB vs FLAN
    if best_doc and confidence >= 0.75:
        answer_en = best_doc["answer"]
        src_context = f"Q: {best_doc['question']}\nA: {best_doc['answer']}"
        source_type = "KB"
    else:
        answer_en = generate_with_flan(query_en, kb_results)
        src_context = ""
        source_type = "FLAN" if FALLBACK_MODEL else "NONE"

    final_answer = translate(answer_en, "en", user_lang)

    # Append and show
    st.session_state.messages.append({
        "role": "user", "content": user_input, "time": datetime.now().strftime("%H:%M:%S")
    })
    st.session_state.messages.append({
        "role": "assistant", "content": final_answer, "time": datetime.now().strftime("%H:%M:%S"),
        "context": src_context, "source": source_type
    })

    st.markdown("---")
    st.subheader(f"📢 Answer ({'Knowledge Base' if source_type=='KB' else 'AI Fallback'})")
    st.write(final_answer)
    if show_context and src_context:
        st.code(src_context)
    st.markdown("---")
