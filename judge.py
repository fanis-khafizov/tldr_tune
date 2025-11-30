"""
LLM-as-a-Judge script using Qwen3-14B for comparing model outputs.
Compares responses from base model vs LoRA fine-tuned model.

Usage:
    python judge.py --base_results results_base.csv --lora_results results_lora.csv --output judge_results.csv
"""

import argparse
import csv
import json
import os
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Prompt template for LLM-as-a-Judge
JUDGE_PROMPT_TEMPLATE = """You are an expert judge evaluating the quality of TL;DR summaries for Reddit posts.

Your task is to compare two summaries and score each one on a scale of 1-5 based on the following criteria:
- **Accuracy**: Does the summary correctly capture the main point of the post?
- **Conciseness**: Is the summary brief and to the point?
- **Relevance**: Does the summary focus on the most important information?
- **Clarity**: Is the summary easy to understand?

### Original Post:
{original_post}

### Ground Truth Summary:
{ground_truth}

### Summary A (Base Model):
{summary_a}

### Summary B (Fine-tuned Model):
{summary_b}

Please evaluate both summaries and respond in the following JSON format only:
```json
{{
    "score_a": <1-5>,
    "score_b": <1-5>,
    "winner": "<A/B/tie>",
    "reasoning": "<brief explanation of your scores>"
}}
```

Respond with only the JSON, no additional text."""


def load_csv_results(path: str) -> list[dict]:
    """Load results from CSV file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def extract_post_from_prompt(prompt: str) -> str:
    """
    Extract the original Reddit post from the Alpaca-style prompt.
    
    The prompt format is:
    ### Instruction:
    Summarize the following Reddit post...
    
    ### Input:
    {post}
    
    ### Response:
    """
    try:
        # Find the content between "### Input:" and "### Response:"
        input_marker = "### Input:"
        response_marker = "### Response:"
        
        start = prompt.find(input_marker)
        end = prompt.find(response_marker)
        
        if start != -1 and end != -1:
            post = prompt[start + len(input_marker):end].strip()
            return post
        return prompt
    except Exception:
        return prompt


def load_judge_model(
    model_path: str,
    device: str = "cuda",
):
    """
    Load Qwen3-14B model for judging.
    
    Args:
        model_path: Path to Qwen3-14B model
        device: Device to load model on
    
    Returns:
        model, tokenizer
    """
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading judge model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    model.eval()
    print("Judge model loaded successfully!")
    return model, tokenizer


def generate_judgment(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    """Generate judgment from the model."""
    # Format for Qwen3 chat model
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Try to use chat template if available
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking for Qwen3
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    except Exception:
        # Fallback to simple prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,  # Lower temperature for more consistent judgments
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()


def parse_judgment(response: str) -> dict:
    """
    Parse the JSON judgment from the model response.
    
    Returns dict with keys: score_a, score_b, winner, reasoning
    """
    try:
        # Try to find JSON in the response
        # Handle case where response might have markdown code blocks
        response = response.replace("```json", "").replace("```", "").strip()
        
        # Find JSON object in response
        start = response.find("{")
        end = response.rfind("}") + 1
        
        if start != -1 and end > start:
            json_str = response[start:end]
            result = json.loads(json_str)
            
            # Validate and normalize
            return {
                "score_a": int(result.get("score_a", 0)),
                "score_b": int(result.get("score_b", 0)),
                "winner": str(result.get("winner", "unknown")),
                "reasoning": str(result.get("reasoning", "")),
            }
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Warning: Failed to parse judgment: {e}")
    
    # Return default if parsing fails
    return {
        "score_a": 0,
        "score_b": 0,
        "winner": "parse_error",
        "reasoning": f"Failed to parse response: {response[:200]}...",
    }


def run_judging(
    model,
    tokenizer,
    base_results: list[dict],
    lora_results: list[dict],
    show_progress: bool = True,
) -> list[dict]:
    """
    Run LLM-as-a-Judge comparison.
    
    Args:
        model: Judge model
        tokenizer: Tokenizer
        base_results: Results from base model
        lora_results: Results from LoRA model
        show_progress: Show progress bar
    
    Returns:
        List of judgment results
    """
    if len(base_results) != len(lora_results):
        raise ValueError(
            f"Mismatch in number of results: base={len(base_results)}, lora={len(lora_results)}"
        )
    
    judgments = []
    iterator = tqdm(
        zip(base_results, lora_results),
        total=len(base_results),
        desc="Judging comparisons"
    ) if show_progress else zip(base_results, lora_results)
    
    for base_item, lora_item in iterator:
        # Extract original post from prompt
        original_post = extract_post_from_prompt(base_item["input"])
        ground_truth = base_item.get("ground_truth", "N/A")
        summary_a = base_item.get("model_output", "")
        summary_b = lora_item.get("model_output", "")
        
        # Create judge prompt
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            original_post=original_post,
            ground_truth=ground_truth,
            summary_a=summary_a,
            summary_b=summary_b,
        )
        
        # Generate judgment
        response = generate_judgment(model, tokenizer, judge_prompt)
        judgment = parse_judgment(response)
        
        # Add to results
        judgments.append({
            "original_post": original_post[:500],  # Truncate for CSV
            "ground_truth": ground_truth,
            "summary_base": summary_a,
            "summary_lora": summary_b,
            "score_base": judgment["score_a"],
            "score_lora": judgment["score_b"],
            "winner": judgment["winner"],
            "reasoning": judgment["reasoning"],
        })
    
    return judgments


def save_judgments_csv(judgments: list[dict], output_path: str):
    """Save judgments to CSV file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    fieldnames = [
        "original_post", "ground_truth", "summary_base", "summary_lora",
        "score_base", "score_lora", "winner", "reasoning"
    ]
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(judgments)
    
    print(f"Judgments saved to {output_path}")


def compute_statistics(judgments: list[dict]) -> dict:
    """Compute summary statistics from judgments."""
    total = len(judgments)
    if total == 0:
        return {}
    
    base_wins = sum(1 for j in judgments if j["winner"].upper() == "A")
    lora_wins = sum(1 for j in judgments if j["winner"].upper() == "B")
    ties = sum(1 for j in judgments if j["winner"].lower() == "tie")
    
    valid_scores = [j for j in judgments if j["score_base"] > 0 and j["score_lora"] > 0]
    
    avg_score_base = sum(j["score_base"] for j in valid_scores) / len(valid_scores) if valid_scores else 0
    avg_score_lora = sum(j["score_lora"] for j in valid_scores) / len(valid_scores) if valid_scores else 0
    
    return {
        "total_comparisons": total,
        "base_wins": base_wins,
        "lora_wins": lora_wins,
        "ties": ties,
        "avg_score_base": round(avg_score_base, 2),
        "avg_score_lora": round(avg_score_lora, 2),
        "base_win_rate": round(base_wins / total * 100, 1),
        "lora_win_rate": round(lora_wins / total * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge for comparing model outputs")
    
    # Input files
    parser.add_argument(
        "--base_results",
        type=str,
        default="results_base.csv",
        help="Path to CSV file with base model results"
    )
    parser.add_argument(
        "--lora_results",
        type=str,
        default="results_lora.csv",
        help="Path to CSV file with LoRA model results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="judge_results.csv",
        help="Path to output CSV file with judgments"
    )
    
    # Model settings
    parser.add_argument(
        "--judge_model_path",
        type=str,
        default="./Qwen3-14B",
        help="Path to Qwen3-14B judge model"
    )
    
    # Processing options
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to judge")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading base model results from {args.base_results}...")
    base_results = load_csv_results(args.base_results)
    print(f"Loaded {len(base_results)} base model results")
    
    print(f"Loading LoRA model results from {args.lora_results}...")
    lora_results = load_csv_results(args.lora_results)
    print(f"Loaded {len(lora_results)} LoRA model results")
    
    # Limit samples if specified
    if args.max_samples is not None:
        base_results = base_results[:args.max_samples]
        lora_results = lora_results[:args.max_samples]
        print(f"Limited to {len(base_results)} samples")
    
    # Load judge model
    model, tokenizer = load_judge_model(
        model_path=args.judge_model_path,
    )
    
    # Run judging
    print("\nRunning LLM-as-a-Judge comparisons...")
    judgments = run_judging(model, tokenizer, base_results, lora_results)
    
    # Save results
    save_judgments_csv(judgments, args.output)
    
    # Print statistics
    stats = compute_statistics(judgments)
    print("\n" + "=" * 50)
    print("JUDGMENT SUMMARY")
    print("=" * 50)
    print(f"Total comparisons: {stats.get('total_comparisons', 0)}")
    print(f"Base model wins: {stats.get('base_wins', 0)} ({stats.get('base_win_rate', 0)}%)")
    print(f"LoRA model wins: {stats.get('lora_wins', 0)} ({stats.get('lora_win_rate', 0)}%)")
    print(f"Ties: {stats.get('ties', 0)}")
    print(f"Average score (Base): {stats.get('avg_score_base', 0)}/5")
    print(f"Average score (LoRA): {stats.get('avg_score_lora', 0)}/5")
    print("=" * 50)
    
    # Save statistics to JSON
    stats_path = args.output.replace(".csv", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")


if __name__ == "__main__":
    main()
