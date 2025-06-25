import model.model_provider as model_provider

model, tokenizer = model_provider.get_model_and_tokenizer()

question = "What is capital of France?"
inputs = tokenizer(question, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

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

full_output = outputs[0]
generated = tokenizer.decode(full_output[prompt_len:], skip_special_tokens=True)
answer = generated.split("Human:")[0].strip()

print(f"Answer: {answer}")