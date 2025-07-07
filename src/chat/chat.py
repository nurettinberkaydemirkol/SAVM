import generate_lora.generate_lora as generate_lora
import generate_synthetic.generate_synthetic_data as generate_synthetic
import chat.embed as embed
import utils.is_generate_lora as is_generate_lora
from providers.vector_db_provider import VectorDatabaseProvider
import uuid
import traceback
from lora_merger.lora_merger import generate_with_merged_model, merge_multiple_loras_and_save, get_model_and_tokenizer


# This script generates a synthetic question-answer pair using a language model,
id = str(uuid.uuid4())

# embeds it, and stores it in a vector database.
lora_db = VectorDatabaseProvider()
dataset_db = VectorDatabaseProvider()
dataset_json_id = None

lora_db.load_from_file("./db/lora_db")
dataset_db.load_from_file("./db/dataset_db")

# Generate a question using the model
model, tokenizer = get_model_and_tokenizer()

# Example question
question = "weather in istanbul"
print(f"Question: {question}")

# Generate the question vector using the embed model
question_vector = embed.create_vector(question)

k_near_synthetic_data = dataset_db.search(query_vector=question_vector.tolist(), k=1, threshold=0.15)

if k_near_synthetic_data:
    print("near synthetic data:", k_near_synthetic_data)
    dataset_json_file = k_near_synthetic_data[0].get("file_uri", None)
    dataset_json_id = k_near_synthetic_data[0].get("id", None)
    is_generate_lora_bool = is_generate_lora(dataset_json_file)
else:
    dataset_json_file = None
    is_generate_lora_bool = False

k_near_lora_files = lora_db.search(query_vector=question_vector.tolist(), k=3, threshold=0.15)
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


if is_generate_lora_bool == True:
    synthetic_data_path = f"/Users/berkaydemirkol/Documents/GitHub/SAVM/synthetic_data_cluster/data-{id}.jsonl"
    generate_synthetic.generate_synthetic_data(question, 10, output_path=synthetic_data_path)
    
    lora_output_dir = f"../lora_files/lora-{id}"
    print(f"Training LoRA and saving to: {lora_output_dir}")
    generated_lora_file = generate_lora.generate_lora(
        data_file=synthetic_data_path,
        output_dir=lora_output_dir
    )

    vector_np = question_vector.squeeze(0).detach().cpu().numpy()
    lora_db.add_or_update(id, vector_np, generated_lora_file)
    lora_db.save_to_file("./db/lora_db")
else:
    synthetic_data_path = f"/Users/berkaydemirkol/Documents/GitHub/SAVM/synthetic_data_cluster/data-{dataset_json_id}.jsonl"
    generate_synthetic.generate_synthetic_data(question, 10, output_path=synthetic_data_path)
    
ALL_VECTOR_LIST = lora_db.list_ids()
print(ALL_VECTOR_LIST)