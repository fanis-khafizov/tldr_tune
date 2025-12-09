# TL;DR Fine-tuning Pipeline

Fine-tuning Llama 3.1 8B Instruct для задачи суммаризации Reddit постов с использованием LoRA.

## Quick Start

```bash
# 1. Установка зависимостей и загрузка моделей (см. SETUP.md)

# 2. Подготовка датасета
python save_tldr.py

# 3. Fine-tune
CUDA_VISIBLE_DEVICES=0 python -m tune run lora_finetune_single_device --config llama3_1_8B_instruct_lora.yaml

# 4. Inference
CUDA_VISIBLE_DEVICES=0 python inference.py --mode lora --input data_tldr/test.jsonl --output results_lora.jsonl \
    --lora_adapter_path ./outputs/llama3_1_8b_instruct_lora
```

## Структура проекта

```
├── Llama-3.1-8B-Instruct/     # Веса базовой модели
├── Qwen3-32B/                  # Веса модели-судьи
├── data_tldr/                  # Датасет (train/validation/test.jsonl)
├── outputs/                    # Результаты обучения
│   └── llama3_1_8b_instruct_lora/  # LoRA адаптеры
├── save_tldr.py                # Скрипт подготовки датасета
├── inference.py                # Скрипт инференса
├── judge.py                    # LLM-as-a-Judge оценка
└── llama3_1_8B_instruct_lora.yaml  # Конфиг для обучения
```

## Обучение

```bash
CUDA_VISIBLE_DEVICES=0 tune run lora_finetune_single_device --config llama3_1_8B_instruct_lora.yaml
```

LoRA адаптеры сохраняются в `./outputs/llama3_1_8b_instruct_lora/`.

## Inference

### Базовая модель

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --mode base --input data_tldr/test.jsonl --output results_base.jsonl
```

### LoRA модель

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --mode lora --input data_tldr/test.jsonl --output results_lora.jsonl \
    --lora_adapter_path ./outputs/llama3_1_8b_instruct_lora
```

### Ограничение выборки (для тестов)

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --mode lora --input data_tldr/test.jsonl --output results_lora.jsonl \
    --lora_adapter_path ./outputs/llama3_1_8b_instruct_lora_sft --max_samples 10
```

## Оценка (LLM-as-a-Judge)

### Стандартная версия (transformers)

```bash
CUDA_VISIBLE_DEVICES=0 python judge.py --base_results results_base.jsonl --lora_results results_lora.jsonl \
    --output judge_results.jsonl
```

### Быстрая версия (vLLM)

```bash
CUDA_VISIBLE_DEVICES=0 python judge_vllm.py --base_results results_base.jsonl --lora_results results_lora.jsonl \
    --output judge_results.jsonl
```

Опциональные параметры:
- `--max_samples 100` — ограничить количество сэмплов для оценки
- `--batch_size 32` — размер батча для vLLM (по умолчанию 32)
- `--judge_model_path ./Qwen3-32B` — путь к модели-судье

Результаты:
- `judge_results.jsonl` — детальные сравнения
- `judge_results_stats.json` — статистика (win rates, средние оценки)
