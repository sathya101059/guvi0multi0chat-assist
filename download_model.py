# download_model.py
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os, shutil

MODEL_ID   = "Satya28Kanth/guvi_flan_finetuned"
CACHE_ROOT = "./cached_model"         # app.py expects ./cached_model/{tokenizer,model}
TMP_DIR    = os.path.join(CACHE_ROOT, "hub")  # temporary staging area

def download_model():
    os.makedirs(CACHE_ROOT, exist_ok=True)
    print(f"[1/3] Downloading snapshot of: {MODEL_ID}")

    # If the model is private, make sure HF_TOKEN is set in the environment.
    snapshot_dir = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=TMP_DIR,
        local_dir_use_symlinks=False  # ensure real files (no broken links in Docker)
    )
    print(f"    ✓ Snapshot ready at: {snapshot_dir}")

    print("[2/3] Materializing tokenizer/model folders for app.py …")
    tok_out = os.path.join(CACHE_ROOT, "tokenizer")
    mdl_out = os.path.join(CACHE_ROOT, "model")

    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(snapshot_dir)
    tokenizer.save_pretrained(tok_out)
    model.save_pretrained(mdl_out)
    print(f"    ✓ Saved tokenizer → {tok_out}")
    print(f"    ✓ Saved model     → {mdl_out}")

    print("[3/3] Cleaning temporary hub folder …")
    try:
        shutil.rmtree(TMP_DIR)
    except Exception as e:
        print("    (cleanup warning)", e)

    print("✅ Done. Cached at ./cached_model/{tokenizer,model}")

if __name__ == "__main__":
    download_model()
