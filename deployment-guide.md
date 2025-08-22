# Deployment Guide for GUVI Multilingual Chatbot

## 1. Deploy on Hugging Face Spaces

### Prerequisites
- Hugging Face account
- HF API Token (from https://huggingface.co/settings/tokens)

### Steps:
1. Create a new Space at https://huggingface.co/spaces
2. Select SDK:  
   - Use **Gradio** or **Docker** (Streamlit SDK might not be available currently)
3. Upload these files to the Space repo:  
   - `app.py`  
   - `fast_guvi_docs.pkl`  
   - `fast_guvi_index.index`  
   - `requirements.txt`  
   - `README.md`
4. In Space **Settings → Secrets**, add your HF_TOKEN=your_huggingface_token

5. Wait for build and deployment to complete.
6. Access your app via the provided URL and test.

---

## 2. Deploy on Streamlit Community Cloud

### Prerequisites
- GitHub account linked with Streamlit Cloud
- Public GitHub repository containing the project

### Steps:
1. Push your project to GitHub.
2. Log in to https://streamlit.io/cloud
3. Click “New app” → select your GitHub repo and branch (`main`)
4. Enter `app.py` as the main file.
5. Add your HF token secret in Streamlit Cloud dashboard → Settings → Secrets:  
6. Deploy and use the live URL for demo and submission.

---

## 3. Local Run

### Prerequisites
- Python 3.8+
- Virtual environment preferred

### Steps:
1. Clone or download the repo.
2. Create and activate a virtual environment (recommended):
python -m venv venv
source venv/bin/activate # Linux/macOS
venv\Scripts\activate # Windows
3. Install dependencies:
pip install -r requirements.txt
4. Export your HF token (keep the terminal open):
export HF_TOKEN="your_huggingface_token"
5. Run the app:
streamlit run app.py

---

## 4. Updating Your Deployment

- Push any code or data changes to GitHub or upload new files on HF Spaces.
- The deployment will automatically rebuild and redeploy.

---

## 5. Troubleshooting

- **Model download errors:** Verify `HF_TOKEN` is set correctly.
- **Build failures:** Check `requirements.txt` includes all needed packages.
- **Missing files:** Ensure all required files are uploaded.
- **Slow startup:** Large models may take longer to download on first run.

---

# End of Deployment Guide



