import json
import os
import providers.model_provider as model_provider
from datetime import datetime

def generate_qa_dataset(topic="Machine Learning basics", num_examples=10, output_path=None):
    model, tokenizer = model_provider.get_model_and_tokenizer()

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = topic.lower().replace(" ", "_").replace("/", "_")
        output_path = f"{safe_topic}_qa_{timestamp}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for idx in range(1, num_examples + 1):
            question_prompt = f"Write a short question about the topic: {topic}"
            inputs = tokenizer(question_prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[-1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                top_k=40
            )

            full_output = outputs[0]
            generated = tokenizer.decode(full_output[prompt_len:], skip_special_tokens=True)
            question = generated.split("Human:")[0].strip()

            answer_prompt = f"Answer briefly and clearly: {question}"
            inputs = tokenizer(answer_prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[-1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_k=40
            )

            full_output = outputs[0]
            generated = tokenizer.decode(full_output[prompt_len:], skip_special_tokens=True)
            answer = generated.split("Human:")[0].strip()

            print(f"[{idx}] Q: {question}\n    A: {answer}")

            json_obj = {
                "prompt": question,
                "answer": answer
            }

            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    print(f"\nAll prompt-answer pairs saved to: {output_path}")
    return output_path