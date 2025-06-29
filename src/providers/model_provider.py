from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv

load_dotenv()

model = None
tokenizer = None

def get_model_and_tokenizer():
    global model, tokenizer
    if model is None or tokenizer is None:
        local_model_path = "/Users/berkaydemirkol/Documents/GitHub/SAVM/ai_models/Qwen"
        print(local_model_path)

        print("Tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
        print("Tokenizer ready.")

        print("Model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model ready.")
        
        return model, tokenizer
