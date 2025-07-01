from transformers import AutoModelForCausalLM
import torch

def check_model_modules(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Check what modules are available in the TinyLlama model"""
    print(f"Loading model: {model_name}")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    print("\nChecking for attention-related modules...")
    
    # Get all module names
    module_names = []
    for name, module in model.named_modules():
        module_names.append(name)
    
    # Look for attention-related modules
    attention_modules = []
    for name in module_names:
        if 'attn' in name.lower() or 'proj' in name.lower():
            attention_modules.append(name)
    
    print(f"\nFound {len(attention_modules)} attention/projection related modules:")
    for module in sorted(attention_modules):
        print(f"  - {module}")
    
    # Specifically check for c_attn and c_proj
    c_attn_found = any('c_attn' in name for name in module_names)
    c_proj_found = any('c_proj' in name for name in module_names)
    
    print(f"\nSpecific module check:")
    print(f"  c_attn found: {c_attn_found}")
    print(f"  c_proj found: {c_proj_found}")
    
    if c_attn_found and c_proj_found:
        print("\n✅ Both c_attn and c_proj modules are available!")
        print("The default target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'] should work fine.")
    else:
        print("\n❌ One or both modules not found.")
        print("You may need to adjust the target_modules parameter.")
        
        # Suggest alternative modules
        print("\nSuggested alternative target modules:")
        alternative_modules = []
        for name in module_names:
            if any(keyword in name.lower() for keyword in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
                alternative_modules.append(name)
        
        if alternative_modules:
            print("  Alternative modules found:")
            for module in sorted(set(alternative_modules)):
                print(f"    - {module}")
        else:
            print("  No obvious alternative modules found.")
    
    return c_attn_found, c_proj_found

if __name__ == "__main__":
    check_model_modules() 