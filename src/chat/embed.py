from transformers import AutoTokenizer, AutoModel
import torch
import os
from dotenv import load_dotenv

load_dotenv()

model_path = os.getenv("EMBED_MODEL_PATH")

def create_vector(text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding_vector = outputs.last_hidden_state[:, 0, :] 

    print("Embedding shape:", embedding_vector.shape)
    print("Embedding vector:", embedding_vector)
    
    return embedding_vector