"""
LLM-as-a-Judge script using Qwen3-32B via vLLM for comparing model outputs.
Fast batch inference for comparing base model vs LoRA fine-tuned model.

Usage:
    python judge_vllm.py --base_results results_base.jsonl --lora_results results_lora.jsonl --output judge_results.jsonl
"""

import argparse
import json
import os

from tqdm import tqdm
from vllm import LLM, SamplingParams


# Prompt template for LLM-as-a-Judge
# Note: Qwen3 will use thinking mode (<think>...</think>) which improves judgment quality
# The parser will extract JSON after the thinking block
JUDGE_PROMPT_TEMPLATE = """You are an expert judge evaluating the quality of TL;DR summaries for Reddit posts.

Your task is to compare two summaries and score each one on a scale of 1-5 based on the following criteria:
- Accuracy: Does the summary correctly capture the main point of the post?
- Conciseness: Is the summary brief and to the point?
- Relevance: Does the summary focus on the most important information?
- Clarity: Is the summary easy to understand?

Original Post:
{original_post}

Ground Truth Summary:
{ground_truth}

Summary A (Base Model):
{summary_a}

Summary B (Fine-tuned Model):
{summary_b}

After your analysis, respond with JSON in this exact format:
{{"score_a": <1-5>, "score_b": <1-5>, "winner": "<A|B|tie>", "reasoning": "<brief explanation>"}}"""


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


def format_judge_prompt(original_post: str, ground_truth: str, summary_a: str, summary_b: str) -> str:
    """Format the judge prompt with Qwen3 chat template."""
    user_content = JUDGE_PROMPT_TEMPLATE.format(
        original_post=original_post,
        ground_truth=ground_truth,
        summary_a=summary_a,
        summary_b=summary_b,
    )
    # Qwen3 chat format
    return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def parse_judgment(response: str) -> dict:
    """Parse the JSON judgment from the model response."""
    try:
        # Clean response
        response = response.strip()
        
        # Remove Qwen3 thinking tags if present
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove common markdown patterns
        for pattern in ["```json", "```", "```\n"]:
            response = response.replace(pattern, "")
        
        # Remove leading/trailing whitespace and newlines
        response = response.strip()
        
        # Find JSON object boundaries
        start = response.find("{")
        end = response.rfind("}") + 1
        
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in response")
        
        json_str = response[start:end]
        result = json.loads(json_str)
        
        # Validate and convert fields
        score_a = int(result.get("score_a", 0))
        score_b = int(result.get("score_b", 0))
        
        # Validate scores are in range
        if not (1 <= score_a <= 5):
            score_a = max(1, min(5, score_a))
        if not (1 <= score_b <= 5):
            score_b = max(1, min(5, score_b))
        
        winner = str(result.get("winner", "unknown")).upper().strip()
        if winner not in ["A", "B", "TIE"]:
            # Infer winner from scores
            if score_a > score_b:
                winner = "A"
            elif score_b > score_a:
                winner = "B"
            else:
                winner = "TIE"
        
        return {
            "score_a": score_a,
            "score_b": score_b,
            "winner": winner,
            "reasoning": str(result.get("reasoning", "")).strip(),
        }
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        print(f"Warning: Failed to parse judgment: {e}")
        print(f"Response: {response[:300]}...")
    
    return {
        "score_a": 0,
        "score_b": 0,
        "winner": "parse_error",
        "reasoning": f"Failed to parse response",
    }


def compute_statistics(judgments: list[dict]) -> dict:
    """Compute summary statistics from judgments."""
    total = len(judgments)
    if total == 0:
        return {}
    
    base_wins = sum(1 for j in judgments if j["winner"].upper() == "A")
    lora_wins = sum(1 for j in judgments if j["winner"].upper() == "B")
    ties = sum(1 for j in judgments if j["winner"].lower() == "tie")
    parse_errors = sum(1 for j in judgments if j["winner"] == "parse_error")
    
    valid_scores = [j for j in judgments if j["score_base"] > 0 and j["score_lora"] > 0]
    
    avg_score_base = sum(j["score_base"] for j in valid_scores) / len(valid_scores) if valid_scores else 0
    avg_score_lora = sum(j["score_lora"] for j in valid_scores) / len(valid_scores) if valid_scores else 0
    
    return {
        "total_comparisons": total,
        "base_wins": base_wins,
        "lora_wins": lora_wins,
        "ties": ties,
        "parse_errors": parse_errors,
        "avg_score_base": round(avg_score_base, 2),
        "avg_score_lora": round(avg_score_lora, 2),
        "base_win_rate": round(base_wins / total * 100, 1),
        "lora_win_rate": round(lora_wins / total * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge via vLLM for comparing model outputs")
    
    # Input files
    parser.add_argument("--base_results", type=str, default="results_base.jsonl",
                        help="Path to JSONL file with base model results")
    parser.add_argument("--lora_results", type=str, default="results_lora.jsonl",
                        help="Path to JSONL file with LoRA model results")
    parser.add_argument("--output", type=str, default="judge_results.jsonl",
                        help="Path to output JSONL file with judgments")
    
    # Model settings
    parser.add_argument("--judge_model_path", type=str, default="./Qwen3-32B",
                        help="Path to Qwen3-32B judge model")
    
    # Processing options
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to judge")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for vLLM inference")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading base model results from {args.base_results}...")
    base_results = load_jsonl(args.base_results)
    print(f"Loaded {len(base_results)} base model results")
    
    print(f"Loading LoRA model results from {args.lora_results}...")
    lora_results = load_jsonl(args.lora_results)
    print(f"Loaded {len(lora_results)} LoRA model results")
    
    if len(base_results) != len(lora_results):
        raise ValueError(f"Mismatch: base={len(base_results)}, lora={len(lora_results)}")
    
    # Limit samples if specified
    if args.max_samples is not None:
        base_results = base_results[:args.max_samples]
        lora_results = lora_results[:args.max_samples]
        print(f"Limited to {len(base_results)} samples")
    
    # Load vLLM model
    print(f"Loading judge model from {args.judge_model_path}...")
    llm = LLM(
        model=args.judge_model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
    )
    
    sampling_params = SamplingParams(
        temperature=0.6,  # Qwen3: 0.5-0.7 for thinking mode
        top_p=0.95,
        max_tokens=2048,  # Enough for thinking + JSON output
        stop=["<|im_end|>"],
    )
    
    # Prepare all prompts
    print("Preparing judge prompts...")
    prompts = []
    items_data = []
    
    for base_item, lora_item in zip(base_results, lora_results):
        original_post = base_item.get("post", "")
        ground_truth = base_item.get("ground_truth", "N/A")
        summary_a = base_item.get("model_output", "")
        summary_b = lora_item.get("model_output", "")
        
        prompt = format_judge_prompt(original_post, ground_truth, summary_a, summary_b)
        prompts.append(prompt)
        items_data.append({
            "original_post": original_post,
            "ground_truth": ground_truth,
            "summary_base": summary_a,
            "summary_lora": summary_b,
        })
    
    # Run batch inference
    print(f"Running batch inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Process results
    judgments = []
    for item_data, output in tqdm(zip(items_data, outputs), total=len(outputs), desc="Parsing judgments"):
        response = output.outputs[0].text
        judgment = parse_judgment(response)
        
        judgments.append({
            **item_data,
            "score_base": judgment["score_a"],
            "score_lora": judgment["score_b"],
            "winner": judgment["winner"],
            "reasoning": judgment["reasoning"],
        })
    
    # Save results
    save_jsonl(judgments, args.output)
    
    # Print statistics
    stats = compute_statistics(judgments)
    print("\n" + "=" * 50)
    print("JUDGMENT SUMMARY")
    print("=" * 50)
    print(f"Total comparisons: {stats.get('total_comparisons', 0)}")
    print(f"Base model wins: {stats.get('base_wins', 0)} ({stats.get('base_win_rate', 0)}%)")
    print(f"LoRA model wins: {stats.get('lora_wins', 0)} ({stats.get('lora_win_rate', 0)}%)")
    print(f"Ties: {stats.get('ties', 0)}")
    print(f"Parse errors: {stats.get('parse_errors', 0)}")
    print(f"Average score (Base): {stats.get('avg_score_base', 0)}/5")
    print(f"Average score (LoRA): {stats.get('avg_score_lora', 0)}/5")
    print("=" * 50)
    
    # Save statistics
    stats_path = args.output.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")


if __name__ == "__main__":
    main()
