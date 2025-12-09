from datasets import load_dataset
import json, os

from prompts import format_prompt, format_output

ds = load_dataset("trl-lib/tldr")


def map_row(r):
    post = r.get("prompt")
    summary = r.get("completion")
    if not post or not summary:
        return None
    return {"input": format_prompt(post), "output": format_output(summary)}


base = "data_tldr"
os.makedirs(base, exist_ok=True)
splits = {}

for split in ["train", "validation", "test"]:
    if split in ds:
        rows = [map_row(r) for r in ds[split] if map_row(r)]
        out_path = os.path.join(base, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        splits[split] = out_path

print("Saved splits:", json.dumps(splits, indent=2))