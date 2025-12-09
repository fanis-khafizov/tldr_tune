"""
Inference script for Llama 3.1-8B-Instruct with torchtune.
Supports base model and LoRA fine-tuned model.
"""

import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from torchtune.models.llama3_1 import llama3_1_8b, lora_llama3_1_8b
from torchtune.models.llama3 import llama3_tokenizer
from torchtune.training import FullModelHFCheckpointer
from torchtune.generation import generate

from prompts import format_prompt


def parse_output(text: str) -> str:
    """Extract clean text before any stop token."""
    # Stop tokens for Llama 3.1 Instruct
    stop_markers = ["<|eot_id|>", "<|end_of_text|>"]
    
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0]
    
    return text.strip()


# ============================================================================
# Data loading
# ============================================================================

def load_data(path: str) -> list[dict]:
    """Load test data, extracting original posts and ground truth."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # Extract post from formatted input
            match = re.search(
                r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.+?)<\|eot_id\|>",
                item["input"],
                re.DOTALL
            )
            post = match.group(1).strip() if match else item["input"]
            ground_truth = parse_output(item.get("output", ""))
            
            data.append({"post": post, "ground_truth": ground_truth})
    return data


def save_jsonl(data: list[dict], path: str):
    """Save results to JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================================
# Model loading
# ============================================================================

CHECKPOINT_FILES = [
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
]


def load_model(
    base_model_path: str,
    lora_adapter_path: str | None = None,
    device: str = "cuda",
):
    """Load Llama model and tokenizer."""
    # Tokenizer
    tokenizer_path = os.path.join(base_model_path, "original", "tokenizer.model")
    tokenizer = llama3_tokenizer(path=tokenizer_path)
    
    # Model
    if lora_adapter_path:
        model = lora_llama3_1_8b(
            lora_attn_modules=["q_proj", "v_proj", "output_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=False,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.0,
        )
        adapter_path = os.path.join(lora_adapter_path, "epoch_0", "adapter_model.safetensors")
        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=base_model_path,
            checkpoint_files=CHECKPOINT_FILES,
            adapter_checkpoint=adapter_path,
            recipe_checkpoint=None,
            output_dir=lora_adapter_path,
            model_type="LLAMA3",
        )
    else:
        model = llama3_1_8b()
        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=base_model_path,
            checkpoint_files=CHECKPOINT_FILES,
            recipe_checkpoint=None,
            output_dir="./outputs/tmp",
            model_type="LLAMA3",
        )
    
    # Load weights
    checkpoint = checkpointer.load_checkpoint()
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    
    return model, tokenizer


# ============================================================================
# Generation
# ============================================================================

def generate_summary(
    model,
    tokenizer,
    post: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cuda",
) -> str:
    """Generate summary for a single post."""
    prompt = format_prompt(post)
    tokens = tokenizer.encode(prompt, add_bos=False, add_eos=False)
    prompt_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Stop tokens: <|eot_id|> (128009) and <|end_of_text|> (128001)
    stop_tokens = tokenizer.stop_tokens
    
    with torch.no_grad():
        output_tokens, _ = generate(
            model=model,
            prompt=prompt_tensor,
            max_generated_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            pad_id=tokenizer.pad_id,
            stop_tokens=stop_tokens,
        )
    
    # Decode only generated tokens (after prompt)
    generated = output_tokens[0, len(tokens):].tolist()
    
    # Remove stop token if it's at the end
    for stop_id in stop_tokens:
        if generated and generated[-1] == stop_id:
            generated = generated[:-1]
    
    # Decode to text
    text = tokenizer.decode(generated)
    
    return parse_output(text)


def run_inference(
    model,
    tokenizer,
    data: list[dict],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    device: str = "cuda",
) -> list[dict]:
    """Run inference on all examples."""
    results = []
    for item in tqdm(data, desc="Generating"):
        summary = generate_summary(
            model, tokenizer, item["post"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        results.append({
            "post": item["post"],
            "model_output": summary,
            "ground_truth": item["ground_truth"],
        })
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Llama 3.1-8B-Instruct Inference")
    parser.add_argument("--mode", type=str, choices=["base", "lora"], required=True)
    parser.add_argument("--input", type=str, default="data_tldr/test.jsonl")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default="./Llama-3.1-8B-Instruct")
    parser.add_argument("--lora_adapter_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"results_{args.mode}.jsonl"
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = load_data(args.input)
    if args.max_samples:
        data = data[:args.max_samples]
    print(f"Loaded {len(data)} examples")
    
    # Load model
    print(f"Loading model ({args.mode} mode)...")
    lora_path = args.lora_adapter_path if args.mode == "lora" else None
    model, tokenizer = load_model(args.base_model_path, lora_path, args.device)
    
    # Run inference
    print("Running inference...")
    results = run_inference(
        model, tokenizer, data,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    
    # Save results
    save_jsonl(results, args.output)
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
