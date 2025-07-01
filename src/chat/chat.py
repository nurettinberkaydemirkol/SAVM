import providers.model_provider as model_provider
import generate_lora.generate_lora as generate_lora
import generate_synthetic.generate_synthetic_data as generate_synthetic
import chat.embed as embed
from providers.vector_db_provider import VectorDatabaseProvider
import uuid

from lora_merger.lora_merger import create_ensemble_model_from_search_results, generate_with_ensemble, create_sequential_ensemble

import torch

# This script generates a synthetic question-answer pair using a language model,
id = str(uuid.uuid4())

# embeds it, and stores it in a vector database.
db = VectorDatabaseProvider()

db.load_from_file("./vector_db")

# Generate a question using the model
model, tokenizer = model_provider.get_model_and_tokenizer()

# Example question
question = "What is capital of France?"
inputs = tokenizer(question, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate the question vector using the embed model
question_vector = embed.create_vector(question)

# search vector
k_near_lora_files = db.search(query_vector=question_vector.tolist(), k=3)
print("k-nearest lora files:")
print(k_near_lora_files)

print("\n=== Merged Ensemble ===")
try:
    local_model_path = "distilgpt2"
    ensemble_model, tokenizer = create_ensemble_model_from_search_results(k_near_lora_files)
    answer = generate_with_ensemble(question)
    print(f"Ensemble Answer: {answer}")
except Exception as e:
    print("Ensemble method failed.")
    traceback.print_exc()

print("\n=== Generating Synthetic Data ===")
generate_synthetic.generate_synthetic_data(question, 10, output_path=f'/Users/berkaydemirkol/Documents/GitHub/SAVM/synthetic_data_cluster/data-{id}.jsonl')

generated_lora_file = generate_lora.generate_lora(
    data_file=f'/Users/berkaydemirkol/Documents/GitHub/SAVM/synthetic_data_cluster/data-{id}.jsonl',
    output_dir=f'../lora_files/lora-{id}'
)

print("question vector:", question_vector)

vector_np = question_vector.squeeze(0).detach().cpu().numpy()
db.add_or_update(id, vector_np, generated_lora_file)
db.save_to_file("./vector_db")

ALL_VECTOR_LIST = db.list_ids()
print(ALL_VECTOR_LIST)