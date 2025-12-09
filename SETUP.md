# Setup Guide

## 1. Создание виртуального окружения

```bash
# С использованием uv (рекомендуется)
uv venv --python 3.12
source .venv/bin/activate

# Или стандартный venv
python3.12 -m venv .venv
source .venv/bin/activate
```

## 2. Установка зависимостей

```bash
uv pip install -e .
```

Или через pip:

```bash
pip install -e .
```

## 3. Flash Attention (опционально, для ускорения)

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
uv pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
rm flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

## 4. Загрузка моделей

### Авторизация в Hugging Face

```bash
hf auth login
```

### Llama 3.1 8B Instruct (для fine-tuning)

```bash
hf download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir ./Llama-3.1-8B-Instruct \
    --exclude "original/consolidated*"
```

### Qwen3-32B (для LLM-as-a-Judge)

```bash
hf download Qwen/Qwen3-32B \
    --local-dir ./Qwen3-32B
```

## 5. Настройка W&B (опционально)

```bash
wandb login
```

Для отключения W&B логирования замените в конфиге:

```yaml
metric_logger:
  _component_: torchtune.training.metric_logging.DiskLogger
  log_dir: ${output_dir}/logs
```

## 6. Проверка установки

```bash
# Проверка torchtune
tune --help

# Проверка CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Проверка моделей
ls -la Llama-3.1-8B-Instruct/
ls -la Qwen3-32B/
```

## Структура после установки

```
├── .venv/                      # Виртуальное окружение
├── Llama-3.1-8B-Instruct/      # ~16GB
│   ├── model-*.safetensors
│   ├── tokenizer.json
│   └── original/
│       └── tokenizer.model
├── Qwen3-32B/                  # ~64GB
│   └── model-*.safetensors
└── ...
```
