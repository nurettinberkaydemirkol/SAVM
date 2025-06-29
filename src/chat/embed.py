from transformers import AutoTokenizer, AutoModel
import torch
import os
from dotenv import load_dotenv

load_dotenv()

model_path = os.getenv("EMBED_MODEL_PATH")

def create_vector(text: str):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Move inputs to the same device as the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the embedding vector (using the first token's hidden state)
    embedding_vector = outputs.last_hidden_state[:, 0, :] 

    print("Embedding shape:", embedding_vector.shape)
    print("Embedding vector:", embedding_vector)
    
    return embedding_vector