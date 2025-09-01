GUVI Multilingual GPT Chatbot

Streamlit App with Hugging Face Deployment

✅ Project Overview

The GUVI Multilingual GPT Chatbot enables learners to interact with an AI assistant in multiple Indian languages. It automatically detects the input language, translates the query into English, retrieves relevant answers from a knowledge base or generates responses using a FLAN-T5 model, and translates the response back to the user's language.

The chatbot is deployed on Hugging Face Spaces and includes:
✔ Real-time translation
✔ Knowledge base search using FAISS
✔ Fallback to FLAN-T5 for generative answers
✔ Interactive Streamlit UI

🌐 Live Demo

👉 Click here to try the chatbot on Hugging Face Spaces

https://huggingface.co/spaces/Satya28Kanth/guvimultilingualbot1

🎥 Demo Video

[Watch the demo video on Google Drive]https://drive.google.com/file/d/1kGfphQhkFXOBN8pZhjeVp35dHA1TPAWx/view?usp=drive_link

📌 Features

✔ Auto language detection (supports English + 8 Indian languages)
✔ Context-aware responses via FAISS Knowledge Base
✔ Fallback response generation with FLAN-T5
✔ Translation with Google Translator API
✔ Responsive Streamlit interface with sidebar controls
✔ Debug panel & source context visibility

🛠 Tech Stack

UI Framework: Streamlit

Models:

sentence-transformers/all-MiniLM-L6-v2 for embeddings

google/flan-t5-small for generative fallback

Knowledge Base: FAISS index + Pickled docs

Translation: deep-translator

Language Detection: langdetect

Deployment: Hugging Face Spaces

Programming Language: Python 3.10+

🔍 Workflow

User enters a query in any supported language.

Language is auto-detected (or user-selected).

Query translated to English using Google Translator.

Search knowledge base (FAISS):

If high-confidence match found → return KB answer.

Else → generate response using FLAN-T5.

Translate answer back to original language.

Display conversation history in a chat-style UI.

✅ Languages Supported

English (en)

Hindi (hi)

Tamil (ta)

Telugu (te)

Kannada (kn)

Malayalam (ml)

Gujarati (gu)

Punjabi (pa)

Marathi (mr)

📦 Setup Instructions
1. Clone the repository
git clone https://github.com/sathya101059/guvi0multi0chat-assist.git
cd guvi0multi0chat-assist

2. Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

3. Install dependencies
pip install -r requirements.txt

4. Run the app locally
streamlit run app.py

⚙ Model & Files Required

Ensure these files are available in the working directory (or Hugging Face Space):

FAISS Index: fast_guvi_index (1).index

Pickled Docs: fast_guvi_docs (1).pkl

Embeddings: kb_embeddings.npy

✅ Deployment

Platform: Hugging Face Spaces

Hardware: CPU (suitable for flan-t5-small)

Cache Optimization:

CACHE_DIR = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR


Startup Command:

streamlit run app.py --server.port 7860 --server.address 0.0.0.0

🧠 Model Details

Base GPT Model: google/flan-t5-small (Generative Fallback)

Embedding Model: sentence-transformers/all-MiniLM-L6-v2

Fine-tuning Scale: Medium-scale (100k–500k tokens)

✅ Project Deliverables

✔ app.py – Streamlit interface
✔ embed.py – Embedding logic
✔ kb_embeddings.npy – Knowledge base embeddings
✔ requirements.txt – Python dependencies
✔ Dockerfile – Container setup
✔ README.md – Documentation
✔ Demo Video – Linked above

✅ Evaluation Checklist

✔ Public GitHub repo with code and README
✔ Hugging Face deployment link
✔ Demo video link
✔ Modular & PEP8-compliant code
✔ Setup instructions included

🚀 Future Enhancements

Add voice input/output support

Integrate multilingual speech-to-text

Expand KB with more GUVI content
