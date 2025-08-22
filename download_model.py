from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def download_model():
    model_name = "Satya28Kanth/guvi_flan_finetuned"

    print(f"Downloading and caching model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Save to local folder for caching during Docker build
    tokenizer.save_pretrained("./cached_model/tokenizer")
    model.save_pretrained("./cached_model/model")
    print("Model cached locally in ./cached_model folder.")

if __name__ == "__main__":
    download_model()
