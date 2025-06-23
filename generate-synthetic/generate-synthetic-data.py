import json
import model.model_provider as model_provider

_model, _tokenizer = model_provider.get_model_and_tokenizer()

num_examples = 10
topic = "Machine Learning basics"
output_file = "structured_json_outputs.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for idx in range(1, num_examples + 1):
        # Generate questioons about the topic
        question_prompt = f"Write a short question about the topic: {topic}"
        inputs = _tokenizer(question_prompt, return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

        outputs = _model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.8,
            top_k=40
        )

        question = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print(f"[{idx}] Q: {question}")

        # Generate answer for the question
        answer_prompt = f"Answer briefly and clearly: {question}"
        inputs = _tokenizer(answer_prompt, return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

        outputs = _model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            top_k=40
        )

        answer = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print(f"[{idx}] A: {answer}")

        # JSON olarak yaz
        json_obj = {
            "prompt": question,
            "answer": answer
        }

        f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print(f"\nAll prompt-answer pairs saved to: {output_file}")