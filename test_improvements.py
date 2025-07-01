#!/usr/bin/env python3
"""
Test script to verify the improvements to the LoRA ensemble system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from generate_synthetic.generate_synthetic_data import generate_synthetic_data
from generate_lora.generate_lora import generate_lora
from lora_merger.lora_merger import create_ensemble_model_from_search_results, generate_with_ensemble
import tempfile
import shutil

def test_improvements():
    print("Testing LoRA ensemble improvements...")
    
    # Test 1: Generate better synthetic data
    print("\n1. Testing improved synthetic data generation...")
    temp_dir = tempfile.mkdtemp()
    data_file = os.path.join(temp_dir, "test_data.jsonl")
    
    try:
        generate_synthetic_data("What is the capital of France?", 5, data_file)
        
        # Check the generated data
        with open(data_file, 'r') as f:
            lines = f.readlines()
            print(f"Generated {len(lines)} training examples:")
            for i, line in enumerate(lines[:3]):  # Show first 3 examples
                import json
                data = json.loads(line)
                print(f"  {i+1}. Prompt: {data['prompt'][:50]}...")
                print(f"     Answer: {data['answer']}")
        
        # Test 2: Generate LoRA with better parameters
        print("\n2. Testing improved LoRA generation...")
        lora_dir = os.path.join(temp_dir, "test_lora")
        generate_lora(data_file, output_dir=lora_dir, epochs=1)  # Use 1 epoch for testing
        
        # Test 3: Test ensemble generation
        print("\n3. Testing improved ensemble generation...")
        
        # Create mock search results
        mock_search_results = [{"file_uri": lora_dir}]
        
        # Test the ensemble
        ensemble_model, tokenizer = create_ensemble_model_from_search_results(mock_search_results)
        answer = generate_with_ensemble("What is the capital of France?")
        print(f"Ensemble answer: {answer}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_improvements() 