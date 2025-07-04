from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
import torch
import os
from dotenv import load_dotenv
import copy
import traceback  

load_dotenv()

model = None
tokenizer = None
base_model = None

def get_model_and_tokenizer():
    global model, tokenizer, base_model
    if model is None or tokenizer is None:
        local_model_path = "distilgpt2"
        print(local_model_path)

        print("Tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        print("Tokenizer ready.")

        print("Base model loading...")
        base_model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        print("Base model ready.")
        
        model = copy.deepcopy(base_model)
        
    return model, tokenizer


def merge_lora_and_save(lora_path, output_path, base_model_path="distilgpt2"):
    """
    Merge a single LoRA adapter into the base model and save the merged model.
    Args:
        lora_path (str): Path to the LoRA adapter directory.
        output_path (str): Directory to save the merged model.
        base_model_path (str): Base model name or path. Default is 'distilgpt2'.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import os

    print(f"Loading base model from: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print(f"Loading LoRA adapter from: {lora_path}")
    lora_model = PeftModel.from_pretrained(model, lora_path)

    print("Merging LoRA weights into base model...")
    merged_model = lora_model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Merge and save complete.")

def generate_with_merged_model(merged_model_path, question, max_new_tokens=150):
    """
    Load a merged model from disk and generate an answer to the question.
    Args:
        merged_model_path (str): Path to the merged model directory.
        question (str): The question to answer.
        max_new_tokens (int): Maximum number of tokens to generate.
    Returns:
        str: The generated answer.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading merged model from: {merged_model_path}")
    model = AutoModelForCausalLM.from_pretrained(merged_model_path)
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)

    prompt = f"Instruction: Answer the following question with a clear, factual response.\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.5,
        top_k=50,
        top_p=0.8,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
    )

    full_output = outputs[0]
    generated = tokenizer.decode(full_output[prompt_len:], skip_special_tokens=True)
    answer = generated.strip()
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    if "Answer:" in answer:
        answer = answer.split("Answer:")[0].strip()
    if "Instruction:" in answer:
        answer = answer.split("Instruction:")[0].strip()
    if answer and not answer.endswith((".", "!", "?")):
        answer = answer + "."
    return answer

def merge_multiple_loras_and_save(lora_paths, output_path, base_model_path="distilgpt2"):
    """
    Merge multiple LoRA adapters into the base model and save the merged model.
    Args:
        lora_paths (list of str): List of paths to LoRA adapter directories.
        output_path (str): Directory to save the merged model.
        base_model_path (str): Base model name or path. Default is 'distilgpt2'.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import os

    print(f"Loading base model from: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    valid_lora_paths = []
    for lora_path in lora_paths:
        if not lora_path or not os.path.exists(lora_path):
            print(f"Invalid lora file: {lora_path}. Continue.")
            continue
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"Merged LoRA from: {lora_path}")
        valid_lora_paths.append(lora_path)

    if not valid_lora_paths:
        print("No lora fliie found to merge. Using base model as merged model.")
        merged_model = model 
    else:
        print("Merging all LoRA weights into base model...")
        merged_model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Merge and save complete.")