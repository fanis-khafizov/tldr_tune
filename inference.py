"""
Inference script for Llama 3.1-8B-Instruct model.
Supports two modes: base model and LoRA fine-tuned model.

Usage:
    python inference.py --mode base --input data_tldr/test.jsonl --output results_base.csv
    python inference.py --mode lora --input data_tldr/test.jsonl --output results_lora.csv
"""

import argparse
import json
import csv
import os
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_jsonl(path: str) -> list[dict]:
    """Load data from JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


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
        model = PeftModel.from_pretrained(model, lora_adapter_path)
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


def save_results_csv(results: list[dict], output_path: str):
    """Save results to CSV file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "model_output", "ground_truth"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_path}")


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
        help="Path to output CSV file (default: results_{mode}.csv)"
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
        args.output = f"results_{args.mode}.csv"
    
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
    save_results_csv(results, args.output)
    
    print(f"\nInference completed!")
    print(f"Mode: {args.mode}")
    print(f"Processed: {len(results)} examples")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
