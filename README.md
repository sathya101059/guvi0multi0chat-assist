# GUVI Multilingual GPT Chatbot

## Project Overview

This project implements a **multilingual GPT-based chatbot** for GUVI. The chatbot can answer queries about **courses, certificates, doubts, and technologies** in multiple Indian languages. It combines:

- **Fine-tuned FLAN model** for instruction-response capabilities
- **FAISS-based retrieval** for knowledge-augmented answers
- **Sentence-Transformer embeddings** for semantic search
- **Streamlit interface** for easy user interaction

---

## Model Details

### FLAN Fine-Tuning
- **Model Used**: `google/flan-t5-small` (CPU-friendly)
- **Scale**: Small-to-Medium, suitable for interactive queries in multiple languages
- **Maximum Tokens**:
  - Input tokens: ~512
  - Output tokens: ~150
- **Languages Supported**:
  - English (`en`)
  - Hindi (`hi`)
  - Tamil (`ta`)
  - Telugu (`te`)
  - Kannada (`kn`)
  - Malayalam (`ml`)
  - Gujarati (`gu`)
  - Punjabi (`pa`)
  - Marathi (`mr`)
- **Usage**: Handles queries that do not match KB confidence threshold. Generates context-aware answers using FLAN.

### FAISS Knowledge Base
- **Purpose**: Retrieve top relevant GUVI knowledge content for queries
- **Index Files**:
  - `fast_guvi_docs (1).pkl` — Preprocessed document embeddings
  - `fast_guvi_index (1).index` — FAISS vector index
  - `kb_embeddings.npy` — Numpy array of embeddings
- **Retrieval Process**:
  1. Encode user query with **SentenceTransformer (`all-MiniLM-L6-v2`)**
  2. FAISS retrieves top-k relevant documents
  3. Retrieved documents are used as answers if score ≥ `CONFIDENCE_THRESHOLD` (0.7)

---

## Project Structure

D:/Final/
├──Dataset GUVI /#Created,Combined  and Trained for the instruction-response capabilities
├── append_fastguvi /#For creating and update the FAISS
├── embed.py # Create/update knowledge embeddings
├── guvi0multi0chat-assist/ # Auxiliary scripts /Folder created as a copy of Github repo
├── guvimultilingualbot1/ # Main app folder/Hugging Face Space folder
│ └── app.py # Streamlit entry point
├── download_model.py # Script to download/setup FLAN model
├── fast_guvi_docs (1).pkl # Preprocessed document embeddings
├── fast_guvi_index (1).index # FAISS index
├── kb_embeddings.npy # Knowledge base embeddings
├── Dockerfile # Optional containerization
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## Installation

1. **Clone the repository**
git clone <your-repo-url>
cd Final
Create virtual environment
python -m venv .venv
Activate virtual environment
.venv\Scripts\activate
Linux / Mac
source .venv/bin/activate
Install dependencies
pip install -r requirements.txt
**Usage**
**Run the Streamlit app**
streamlit run guvimultilingualbot1/app.py
Open browser at http://localhost:8501

Select Language (Auto-detect or manually choose from supported Indian languages)

Ask Questions — the bot will retrieve knowledge from FAISS or generate an answer via FLAN.

#Optional Debug Panel
Shows detected language, translated query, KB hits, and FAISS scores.

#How It Works
User Input → Detect language → Translate to English if needed
KB Search → Query embeddings via SentenceTransformer → FAISS retrieves top results

#Answer Selection:
If KB score ≥ 0.7 → Use KB answer
Else → Pass query to FLAN for generation
Translate Back → If user language is non-English
Display Answer → Streamlit chat interface with optional source context

#Notes
Cache and Models: Writable cache paths used to store Transformers and SentenceTransformer caches

CPU-Friendly: Model choice (flan-t5-small) supports small-to-medium scale token inference (~512 input / 150 output tokens)

Multilingual Support: Fully supports major Indian languages as per INDIAN_LANGUAGES mapping in app.py

Authors
Sathya Praveen Raj