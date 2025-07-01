#!/usr/bin/env python3
"""
Test script to verify the improvements work with dynamic Qwen model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_synthetic_data_generation():
    """Test the improved synthetic data generation"""
    print("Testing improved synthetic data generation...")
    
    try:
        from generate_synthetic.generate_synthetic_data import generate_synthetic_data
        
        # Test with a simple topic
        output_file = generate_synthetic_data("What is the capital of France?", 3)
        
        # Check the generated data
        with open(output_file, 'r') as f:
            lines = f.readlines()
            print(f"Generated {len(lines)} training examples:")
            for i, line in enumerate(lines):
                import json
                data = json.loads(line)
                print(f"  {i+1}. Prompt: {data['prompt']}")
                print(f"     Answer: {data['answer']}")
                print()
        
        print("✅ Synthetic data generation test passed!")
        return output_file
        
    except Exception as e:
        print(f"❌ Synthetic data generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_lora_generation(data_file):
    """Test the improved LoRA generation"""
    if not data_file:
        print("❌ Skipping LoRA test - no data file")
        return None
        
    print("\nTesting improved LoRA generation...")
    
    try:
        from generate_lora.generate_lora import generate_lora
        import tempfile
        
        temp_dir = tempfile.mkdtemp()
        lora_dir = os.path.join(temp_dir, "test_lora")
        
        # Generate LoRA with improved parameters
        lora_path = generate_lora(data_file, output_dir=lora_dir, epochs=1)
        
        print(f"✅ LoRA generation test passed! LoRA saved to: {lora_path}")
        return lora_path
        
    except Exception as e:
        print(f"❌ LoRA generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_ensemble_generation(lora_path):
    """Test the improved ensemble generation"""
    if not lora_path:
        print("❌ Skipping ensemble test - no LoRA path")
        return
        
    print("\nTesting improved ensemble generation...")
    
    try:
        from lora_merger.lora_merger import create_ensemble_model_from_search_results, generate_with_ensemble
        
        # Create mock search results
        mock_search_results = [{"file_uri": lora_path}]
        
        # Test the ensemble
        ensemble_model, tokenizer = create_ensemble_model_from_search_results(mock_search_results)
        
        # Test with the original question
        question = "What is the capital of France?"
        answer = generate_with_ensemble(question)
        
        print(f"Question: {question}")
        print(f"Ensemble Answer: {answer}")
        
        print("✅ Ensemble generation test passed!")
        
    except Exception as e:
        print(f"❌ Ensemble generation test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Testing LoRA ensemble improvements with dynamic Qwen model...")
    
    # Test 1: Synthetic data generation
    data_file = test_synthetic_data_generation()
    
    # Test 2: LoRA generation
    lora_path = test_lora_generation(data_file)
    
    # Test 3: Ensemble generation
    test_ensemble_generation(lora_path)
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main() 