import json
import os
import providers.model_provider as model_provider
from datetime import datetime

def generate_synthetic_data(topic="Machine Learning basics", num_examples=10, output_path=None):
    model, tokenizer = model_provider.get_model_and_tokenizer()

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = topic.lower().replace(" ", "_").replace("/", "_")
        output_path = f"{safe_topic}_qa_{timestamp}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for idx in range(1, num_examples + 1):
            question_prompt = f"Generate a clear, factual question about: {topic}. The question should be specific and answerable. Question:"
            inputs = tokenizer(question_prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[-1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.3,  # Lower temperature for more focused questions
                top_k=20,         # Lower top_k for better quality
                top_p=0.9,        # Add top_p for better sampling
                repetition_penalty=1.1,  # Prevent repetition
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            full_output = outputs[0]
            generated = tokenizer.decode(full_output[prompt_len:], skip_special_tokens=True)
            question = generated.split("Question:")[0].strip()
            
            if question.endswith("?"):
                question = question[:-1].strip()
            question = question + "?"

            answer_prompt = f"Provide a clear, factual, and concise answer to this question: {question}\nAnswer:"
            inputs = tokenizer(answer_prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[-1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.2,  # Very low temperature for factual answers
                top_k=15,         # Lower top_k for more focused answers
                top_p=0.85,       # Lower top_p for better quality
                repetition_penalty=1.2,  # Stronger repetition penalty
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            full_output = outputs[0]
            generated = tokenizer.decode(full_output[prompt_len:], skip_special_tokens=True)
            answer = generated.split("Answer:")[0].strip()
            
            if answer.endswith("."):
                answer = answer[:-1].strip()
            answer = answer + "."

            print(f"[{idx}] Q: {question}\n    A: {answer}")

            json_obj = {
                "prompt": f"Question: {question}\nAnswer:",
                "answer": answer
            }

            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    print(f"\nAll prompt-answer pairs saved to: {output_path}")
    return output_path