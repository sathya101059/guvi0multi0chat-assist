import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

FAISS_DOCS = "fast_guvi_docs (1).pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KB_EMB_FILE = "kb_embeddings.npy"

# Load docs
with open(FAISS_DOCS, "rb") as f:
    docs = pickle.load(f)

# Initialize embedder
embedder = SentenceTransformer(EMBED_MODEL)

# Precompute embeddings
texts = [doc["text"] for doc in docs]
embeddings = embedder.encode(texts, normalize_embeddings=True)

# Save embeddings
np.save(KB_EMB_FILE, embeddings.astype('float32'))
print(f"Saved {KB_EMB_FILE} with shape {embeddings.shape}")
