from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
import torch
import os
from dotenv import load_dotenv
import copy
import traceback  # Added for error handling

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

def load_multiple_loras(lora_paths, base_model=None):
    if base_model is None:
        base_model, _ = get_model_and_tokenizer()
        base_model = globals()['base_model']
    
    local_model_path = "distilgpt2"
    merged_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    for i, lora_path in enumerate(lora_paths):
        if not os.path.exists(lora_path):
            print(f"LoRA path does not exist: {lora_path}")
            continue

        try:
            print(f"Loading LoRA {i+1}/{len(lora_paths)}: {lora_path}")
            
            # Load the adapter
            lora_model = PeftModel.from_pretrained(
                merged_model,
                lora_path,
                torch_dtype=torch.float32
            )
            
            # Merge the adapter weights
            merged_model = lora_model.merge_and_unload()
            
            print(f"Successfully merged LoRA: {lora_path}")
        
        except Exception as e:
            print(f"Error loading LoRA {lora_path}: {e}")
            traceback.print_exc()
            continue

    return merged_model

def create_ensemble_model_from_search_results(search_results):
    """
    Create an ensemble model from vector database search results
    search_results: list of tuples (id, vector, lora_file_path)
    """
    global model, tokenizer
    
    lora_paths = [result["file_uri"] for result in search_results if "file_uri" in result]
    
    print(f"Found {len(lora_paths)} LoRA files to merge:")
    for path in lora_paths:
        print(f"  - {path}")
    
    base_model, tokenizer = get_model_and_tokenizer()
    base_model = globals()['base_model']
    
    merged_model = load_multiple_loras(lora_paths, base_model)
    
    model = merged_model
    
    return merged_model, tokenizer

def generate_with_ensemble(question, max_new_tokens=150):
    """
    Generate answer using the ensemble model
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise ValueError("Model not loaded. Call create_ensemble_model_from_search_results first.")
    
    # Create a better prompt format with instruction
    prompt = f"Instruction: Answer the following question with a clear, factual response.\n\nQuestion: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    prompt_len = inputs["input_ids"].shape[-1]
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,  # Very low temperature for factual responses
        top_k=10,         # Very low top_k for focused sampling
        top_p=0.8,        # Lower top_p for better quality
        repetition_penalty=1.3,  # Strong repetition penalty
        no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,     # Stop when EOS token is generated
    )
    
    full_output = outputs[0]
    generated = tokenizer.decode(full_output[prompt_len:], skip_special_tokens=True)
    
    # Clean up the response
    answer = generated.strip()
    # Remove any remaining prompt artifacts
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    if "Answer:" in answer:
        answer = answer.split("Answer:")[0].strip()
    if "Instruction:" in answer:
        answer = answer.split("Instruction:")[0].strip()
    
    # Ensure the answer ends with proper punctuation
    if answer and not answer.endswith(('.', '!', '?')):
        answer = answer + '.'
    
    return answer

def create_sequential_ensemble(search_results, question):
    """
    Apply LoRAs sequentially for memory efficiency
    """
    lora_paths = [result[2] for result in search_results if len(result) > 2]
    
    base_model, tokenizer = get_model_and_tokenizer()
    base_model = globals()['base_model']
    
    answers = []
    
    for i, lora_path in enumerate(lora_paths):
        if os.path.exists(lora_path):
            try:
                print(f"Applying LoRA {i+1}/{len(lora_paths)}: {lora_path}")
                
                lora_model = PeftModel.from_pretrained(
                    base_model, 
                    lora_path,
                    torch_dtype=torch.float32
                )
                
                # Create better prompt format with instruction
                prompt = f"Instruction: Answer the following question with a clear, factual response.\n\nQuestion: {question}\nAnswer:"
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(lora_model.device) for k, v in inputs.items()}
                
                prompt_len = inputs["input_ids"].shape[-1]
                
                outputs = lora_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.1,  # Very low temperature for factual responses
                    top_k=10,         # Very low top_k for focused sampling
                    top_p=0.8,        # Lower top_p for better quality
                    repetition_penalty=1.3,  # Strong repetition penalty
                    no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True,     # Stop when EOS token is generated
                )
                
                full_output = outputs[0]
                generated = tokenizer.decode(full_output[prompt_len:], skip_special_tokens=True)
                
                # Clean up the response
                answer = generated.strip()
                if "Question:" in answer:
                    answer = answer.split("Question:")[0].strip()
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[0].strip()
                if "Instruction:" in answer:
                    answer = answer.split("Instruction:")[0].strip()
                
                # Ensure the answer ends with proper punctuation
                if answer and not answer.endswith(('.', '!', '?')):
                    answer = answer + '.'
                
                answers.append(answer)
                
                del lora_model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error with LoRA {lora_path}: {e}")
                continue
    
    return answers