import providers.model_provider as model_provider
import generate_lora.generate_lora as generate_lora
import generate_synthetic.generate_synthetic_data as generate_synthetic
import chat.embed as embed
from providers.vector_db_provider import VectorDatabaseProvider
import uuid
import traceback
import os
from lora_merger.lora_merger import merge_lora_and_save, generate_with_merged_model, merge_multiple_loras_and_save

import torch

# This script generates a synthetic question-answer pair using a language model,
id = str(uuid.uuid4())

# embeds it, and stores it in a vector database.
db = VectorDatabaseProvider()

db.load_from_file("./vector_db")

# Generate a question using the model
model, tokenizer = model_provider.get_model_and_tokenizer()

# Example question
question = "France's capital city"
print(f"Question: {question}")

inputs = tokenizer(question, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate the question vector using the embed model
question_vector = embed.create_vector(question)

k_near_lora_files = db.search(query_vector=question_vector.tolist(), k=3)
print("k-nearest lora files:")
lora_paths = []
for i, result in enumerate(k_near_lora_files):
    lora_path = result.get('file_uri', None)
    print(f"  {i+1}. {lora_path if lora_path else 'No file URI'}")
    if lora_path:
        lora_paths.append(lora_path)

merged_model_dir = f"../merged_models/merged-{id}"
print(f"Merging LoRAs into base model and saving to: {merged_model_dir}")
merge_multiple_loras_and_save(lora_paths, merged_model_dir)

# Generate answer from merged model
print("\n=== Answer from Merged Model ===")
try:
    answer = generate_with_merged_model(merged_model_dir, question)
    print(f"Merged Model Answer: {answer}")
except Exception as e:
    print("Failed to generate answer from merged model.")
    traceback.print_exc()

print("question vector:", question_vector)


print("\n=== Generating Synthetic Data ===")
synthetic_data_path = f"/Users/berkaydemirkol/Documents/GitHub/SAVM/synthetic_data_cluster/data-{id}.jsonl"
generate_synthetic.generate_synthetic_data(question, 10, output_path=synthetic_data_path)

lora_output_dir = f"../lora_files/lora-{id}"
print(f"Training LoRA and saving to: {lora_output_dir}")
generated_lora_file = generate_lora.generate_lora(
    data_file=synthetic_data_path,
    output_dir=lora_output_dir
)

vector_np = question_vector.squeeze(0).detach().cpu().numpy()
db.add_or_update(id, vector_np, generated_lora_file)
db.save_to_file("./vector_db")

ALL_VECTOR_LIST = db.list_ids()
print(ALL_VECTOR_LIST)