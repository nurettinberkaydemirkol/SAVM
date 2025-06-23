import model.model_provider as model_provider

model, tokenizer = model_provider.get_model_and_tokenizer()

questionPrompt = "What is the capital of France?"

inputs = tokenizer(questionPrompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_k=40
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(f"Answer: {answer}")
