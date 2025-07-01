#!/usr/bin/env python3
"""
Example usage of LoRA generation and merging with different models
"""

import generate_lora.generate_lora as generate_lora
from lora_merger.lora_merger import create_ensemble_model_from_search_results, generate_with_ensemble

def example_with_tinyllama():
    """Example using TinyLlama model"""
    print("=== Example with TinyLlama ===")
    
    # Generate LoRA with TinyLlama (auto-detects target modules)
    lora_output = generate_lora.generate_lora(
        data_file="your_data.jsonl",
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir="./lora_tinyllama",
        epochs=2,
        batch_size=1
    )
    
    # Use the LoRA with TinyLlama
    search_results = [{"file_uri": lora_output}]
    ensemble_model, tokenizer = create_ensemble_model_from_search_results(
        search_results, 
        base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    
    answer = generate_with_ensemble("What is machine learning?")
    print(f"Answer: {answer}")

def example_with_gpt2():
    """Example using GPT-2 model"""
    print("\n=== Example with GPT-2 ===")
    
    # Generate LoRA with GPT-2 (auto-detects target modules)
    lora_output = generate_lora.generate_lora(
        data_file="your_data.jsonl",
        base_model="gpt2",
        output_dir="./lora_gpt2",
        epochs=2,
        batch_size=1
    )
    
    # Use the LoRA with GPT-2
    search_results = [{"file_uri": lora_output}]
    ensemble_model, tokenizer = create_ensemble_model_from_search_results(
        search_results, 
        base_model_name="gpt2"
    )
    
    answer = generate_with_ensemble("What is machine learning?")
    print(f"Answer: {answer}")

def example_with_custom_target_modules():
    """Example with custom target modules"""
    print("\n=== Example with Custom Target Modules ===")
    
    # You can also specify target modules manually
    custom_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Only attention layers
    
    lora_output = generate_lora.generate_lora(
        data_file="your_data.jsonl",
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir="./lora_custom",
        epochs=2,
        batch_size=1,
        target_modules=custom_target_modules  # Override auto-detection
    )
    
    print(f"LoRA generated with custom target modules: {custom_target_modules}")

def check_model_compatibility():
    """Check what target modules different models use"""
    print("\n=== Model Compatibility Check ===")
    
    models_to_check = [
        "gpt2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/DialoGPT-medium",
        "EleutherAI/gpt-neo-125M"
    ]
    
    for model_name in models_to_check:
        print(f"\nChecking {model_name}:")
        try:
            target_modules = generate_lora.get_target_modules_for_model(model_name)
            print(f"  Target modules: {target_modules}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    # Uncomment the examples you want to run
    # example_with_tinyllama()
    # example_with_gpt2()
    # example_with_custom_target_modules()
    
    # This will check what target modules different models use
    check_model_compatibility() 