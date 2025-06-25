from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv

load_dotenv()

_model = None
_tokenizer = None

def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        local_model_path = os.getenv("LOCAL_MODEL_PATH")

        print("Tokenizer loading...")
        _tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
        print("Tokenizer ready.")

        print("Model loading...")
        _model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model ready.")
        
        return _model, _tokenizer
