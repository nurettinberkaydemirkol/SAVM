import providers.model_provider as model_provider
import generate_lora.generate_lora as generate_lora
import chat.embed as embed
from providers.vector_db_provider import VectorDatabaseProvider
import uuid

# This script generates a synthetic question-answer pair using a language model,
id = str(uuid.uuid4())

# embeds it, and stores it in a vector database.
db = VectorDatabaseProvider()

# Generate a question using the model
model, tokenizer = model_provider.get_model_and_tokenizer()

# Example question
question = "What is capital of France?"
inputs = tokenizer(question, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate the question vector using the embed model
question_vector = embed.create_vector(question)

# search vector
k_near_lora_files = db.search(query_vector=question_vector.tolist(), k=5)
print(k_near_lora_files)

# Generate the answer
prompt_len = inputs["input_ids"].shape[-1]

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_k=40,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Decode the generated output
full_output = outputs[0]
generated = tokenizer.decode(full_output[prompt_len:], skip_special_tokens=True)
answer = generated.split("Human:")[0].strip()

print(f"Answer: {answer}")

generated_lora_file = generate_lora.generate_lora(
    data_file="/Users/berkaydemirkol/Documents/GitHub/SAVM/synthetic_data_cluster/data.jsonl",
    output_dir=f'../lora_files/lora-{id}'
)

print("question vector:", question_vector)

vector_np = question_vector.squeeze(0).detach().cpu().numpy()
db.add_or_update(id, vector_np, generated_lora_file)

ALL_VECTOR_LIST = db.list_ids()
print(ALL_VECTOR_LIST)