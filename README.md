# TL;DR Fine-tuning Pipeline

## 1. Download Models

```bash
hf download Qwen/Qwen3-14B --local-dir ./Qwen3-14B
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir ./Llama-3.1-8B-Instruct --exclude "original/consolidated*"
```

## 2. Install Flash Attention (optional)

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
uv pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

## 3. Prepare Dataset

```bash
python save_tldr.py
```

This creates `data_tldr/` directory with `train.jsonl`, `validation.jsonl`, and `test.jsonl`.

## 4. Fine-tune with LoRA

```bash
CUDA_VISIBLE_DEVICES=0 tune run lora_finetune_single_device --config llama3_1_8B_lora_single_device.yaml
```

LoRA adapters will be saved to `./Llama3_1_8B/lora_single_device/`.

## 5. Run Inference

### Base Model

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --mode base --input data_tldr/test.jsonl --output results_base.jsonl
```

### LoRA Fine-tuned Model

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --mode lora --input data_tldr/test.jsonl --output results_lora.jsonl --lora_adapter_path ./Llama3_1_8B/lora_single_device
```

### With sample limit (for testing)

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --mode base --input data_tldr/test.jsonl --output results_base.jsonl --max_samples 100
CUDA_VISIBLE_DEVICES=0 python inference.py --mode lora --input data_tldr/test.jsonl --output results_lora.jsonl --max_samples 100 --lora_adapter_path ./Llama3_1_8B/lora_single_device
```

## 6. Run LLM-as-a-Judge Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python judge.py --base_results results_base.jsonl --lora_results results_lora.jsonl --output judge_results.jsonl
```

### With sample limit

```bash
CUDA_VISIBLE_DEVICES=0 python judge.py --base_results results_base.jsonl --lora_results results_lora.jsonl --output judge_results.jsonl --max_samples 100
```

Results:
- `judge_results.jsonl` — detailed comparison results
- `judge_results_stats.json` — summary statistics (win rates, average scores)
