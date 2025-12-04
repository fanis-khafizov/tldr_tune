"""
Inference script for Llama 3.1-8B-Instruct model using torchtune.
Supports two modes: base model and LoRA fine-tuned model.

Usage:
    python inference.py --mode base --input data_tldr/test.jsonl --output results_base.jsonl
    python inference.py --mode lora --input data_tldr/test.jsonl --output results_lora.jsonl
"""

import argparse
import json
import os
from tqdm import tqdm

import torch
from torchtune.models.llama3_1 import llama3_1_8b, lora_llama3_1_8b
from torchtune.models.llama3 import llama3_tokenizer, Llama3Tokenizer
from torchtune.training import FullModelHFCheckpointer
from torchtune.generation import generate


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


def load_model_and_tokenizer(
    base_model_path: str,
    output_path: str,
    lora_adapter_path: str | None = None,
    device: str = "cuda",
):
    """
    Load Llama model and tokenizer using torchtune.
    
    Args:
        base_model_path: Path to the base Llama model
        lora_adapter_path: Path to LoRA adapters (None for base model only)
        device: Device to load model on
    
    Returns:
        model, tokenizer
    """
    # Load tokenizer
    tokenizer_path = os.path.join(base_model_path, "original", "tokenizer.model")
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = llama3_tokenizer(path=tokenizer_path)
    
    # Checkpoint files
    checkpoint_files = [
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ]
    
    if lora_adapter_path is not None:
        # Load model with LoRA architecture
        print("Creating LoRA model...")
        model = lora_llama3_1_8b(
            lora_attn_modules=['q_proj', 'v_proj', 'output_proj'],
            apply_lora_to_mlp=True,
            apply_lora_to_output=False,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.0,
        )
        
        # Setup checkpointer with adapter
        print(f"Loading base model from {base_model_path}...")
        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=base_model_path,
            checkpoint_files=checkpoint_files,
            adapter_checkpoint=os.path.join(lora_adapter_path, "epoch_0", "adapter_model.safetensors"),
            recipe_checkpoint=None,
            output_dir=output_path,
            model_type="LLAMA3",
        )
    else:
        # Load base model without LoRA
        print("Creating base model...")
        model = llama3_1_8b()
        
        print(f"Loading base model from {base_model_path}...")
        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=base_model_path,
            checkpoint_files=checkpoint_files,
            recipe_checkpoint=None,
            output_dir=output_path,
            model_type="LLAMA3",
        )
    
    # Load weights
    checkpoint_dict = checkpointer.load_checkpoint()
    model.load_state_dict(checkpoint_dict["model"], strict=False)
    
    # Move to device and set dtype
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    # model.compile()
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(
    model,
    tokenizer: Llama3Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cuda",
) -> str:
    """
    Generate response for a given prompt using torchtune generate.
    """
    # Tokenize prompt
    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    prompt_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        generated_tokens, _ = generate(
            model=model,
            prompt=prompt_tensor,
            max_generated_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            pad_id=tokenizer.pad_id,
            stop_tokens=[tokenizer.eos_id]
        )
    
    # Decode only the generated part
    generated_tokens = generated_tokens[0, len(tokens):].tolist()
    response = tokenizer.decode(generated_tokens)
    
    return response.strip()


def run_inference(
    model,
    tokenizer,
    data: list[dict],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    device: str = "cuda",
    show_progress: bool = True,
) -> list[dict]:
    """
    Run inference on a list of examples.
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
            device=device,
        )
        
        results.append({
            "input": prompt,
            "model_output": model_output,
            "ground_truth": ground_truth,
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Llama 3.1-8B Inference Script (torchtune)")
    
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
        default="./Llama-3.1-8B",
        help="Path to base Llama model"
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapters (used only in 'lora' mode)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output_dir",
        help="Path to checkpoint output"
    )
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
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
        output_path=args.output_path,
        lora_adapter_path=lora_path,
        device=args.device,
    )
    
    print(f"\nRunning inference in '{args.mode}' mode...")
    
    # Run inference
    results = run_inference(
        model,
        tokenizer,
        data,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    
    # Save results
    save_jsonl(results, args.output)
    
    print(f"\nInference completed!")
    print(f"Mode: {args.mode}")
    print(f"Processed: {len(results)} examples")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
