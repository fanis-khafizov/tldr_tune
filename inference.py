"""
Inference script for Llama 3.1-8B-Instruct model.
Supports two modes: base model and LoRA fine-tuned model.

Usage:
    python inference.py --mode base --input data_tldr/test.jsonl --output results_base.jsonl
    python inference.py --mode lora --input data_tldr/test.jsonl --output results_lora.jsonl
"""

import argparse
import json
import os
import glob
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file


def load_jsonl(path: str) -> list[dict]:
    """Load data from JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Results saved to {path}")


def load_torchtune_lora_weights(adapter_path: str) -> dict:
    """
    Load LoRA weights saved by torchtune.
    Torchtune saves adapters as safetensors files.
    """
    # Find all safetensors files in the adapter directory
    safetensor_files = glob.glob(os.path.join(adapter_path, "*.safetensors"))
    
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {adapter_path}")
    
    # Load all weights
    all_weights = {}
    for sf_file in safetensor_files:
        print(f"Loading weights from {sf_file}...")
        weights = load_file(sf_file)
        all_weights.update(weights)
    
    return all_weights


def convert_torchtune_to_peft_keys(torchtune_weights: dict) -> dict:
    """
    Convert torchtune LoRA weight keys to PEFT format.
    
    Torchtune format: 'layers.0.attn.q_proj.lora_a.weight'
    PEFT format: 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'
    """
    peft_weights = {}
    
    for key, value in torchtune_weights.items():
        # Skip non-lora weights
        if 'lora_a' not in key and 'lora_b' not in key:
            continue
        
        new_key = key
        
        # Torchtune uses 'attn' but HF uses 'self_attn'
        new_key = new_key.replace(".attn.", ".self_attn.")
        
        # Torchtune uses 'output_proj' but HF uses 'o_proj'
        new_key = new_key.replace(".output_proj.", ".o_proj.")
        
        # Add PEFT prefix
        new_key = "base_model.model.model." + new_key
        
        # Convert lora_a -> lora_A.default, lora_b -> lora_B.default
        new_key = new_key.replace(".lora_a.", ".lora_A.default.")
        new_key = new_key.replace(".lora_b.", ".lora_B.default.")
        
        peft_weights[new_key] = value
    
    return peft_weights


def load_model_and_tokenizer(
    base_model_path: str,
    lora_adapter_path: str | None = None,
    device: str = "cuda",
):
    """
    Load Llama model and tokenizer.
    
    Args:
        base_model_path: Path to the base Llama model
        lora_adapter_path: Path to LoRA adapters (None for base model only)
        device: Device to load model on
    
    Returns:
        model, tokenizer
    """
    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    # Load LoRA adapters if specified
    if lora_adapter_path is not None:
        print(f"Loading LoRA adapters from {lora_adapter_path}...")
        
        # Load torchtune weights
        torchtune_weights = load_torchtune_lora_weights(lora_adapter_path)
        print(f"Loaded {len(torchtune_weights)} weight tensors from torchtune checkpoint")
        
        # Create PEFT LoRA config matching torchtune config
        # From llama3_1_8B_lora_single_device.yaml:
        # lora_attn_modules: ['q_proj', 'v_proj', 'output_proj']
        # apply_lora_to_mlp: True
        # lora_rank: 8
        # lora_alpha: 16
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=[
                "q_proj", "v_proj", "o_proj",  # attention (output_proj = o_proj in HF)
                "gate_proj", "up_proj", "down_proj",  # MLP
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Convert and load weights
        peft_weights = convert_torchtune_to_peft_keys(torchtune_weights)
        print(f"Converted to {len(peft_weights)} PEFT weight tensors")
        
        # Load state dict
        missing, unexpected = model.load_state_dict(peft_weights, strict=False)
        
        if missing:
            # Filter out non-lora missing keys (expected)
            lora_missing = [k for k in missing if 'lora' in k.lower()]
            if lora_missing:
                print(f"Warning: Missing LoRA keys: {len(lora_missing)}")
                for k in lora_missing[:5]:
                    print(f"  - {k}")
        
        if unexpected:
            print(f"Warning: Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"  - {k}")
        
        print("LoRA adapters loaded successfully!")
    
    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    Generate response for a given prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
    
    Returns:
        Generated text (response only, without the prompt)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (excluding the input prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()


def run_inference(
    model,
    tokenizer,
    data: list[dict],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    show_progress: bool = True,
) -> list[dict]:
    """
    Run inference on a list of examples.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        data: List of dicts with 'input' and 'output' (ground truth) keys
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        show_progress: Whether to show progress bar
    
    Returns:
        List of dicts with 'input', 'model_output', 'ground_truth' keys
    """
    results = []
    iterator = tqdm(data, desc="Generating responses") if show_progress else data
    
    for item in iterator:
        prompt = item["input"]
        ground_truth = item.get("output", "")
        
        model_output = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        results.append({
            "input": prompt,
            "model_output": model_output,
            "ground_truth": ground_truth,
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Llama 3.1-8B Inference Script")
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["base", "lora"],
        required=True,
        help="Inference mode: 'base' for base model, 'lora' for LoRA fine-tuned model"
    )
    
    # Paths
    parser.add_argument(
        "--input",
        type=str,
        default="data_tldr/test.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSONL file (default: results_{mode}.jsonl)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./Llama-3.1-8B-Instruct",
        help="Path to base Llama model"
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default="./Llama3_1_8B/lora_single_device",
        help="Path to LoRA adapters (used only in 'lora' mode)"
    )
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    # Set default output path based on mode
    if args.output is None:
        args.output = f"results_{args.mode}.jsonl"
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} examples")
    
    # Limit samples if specified
    if args.max_samples is not None:
        data = data[:args.max_samples]
        print(f"Limited to {len(data)} samples")
    
    # Load model
    lora_path = args.lora_adapter_path if args.mode == "lora" else None
    model, tokenizer = load_model_and_tokenizer(
        base_model_path=args.base_model_path,
        lora_adapter_path=lora_path,
    )
    
    print(f"\nRunning inference in '{args.mode}' mode...")
    
    # Run inference
    results = run_inference(
        model,
        tokenizer,
        data,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    # Save results
    save_jsonl(results, args.output)
    
    print(f"\nInference completed!")
    print(f"Mode: {args.mode}")
    print(f"Processed: {len(results)} examples")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
