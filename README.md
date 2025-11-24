```
hf download Qwen/Qwen3-14B --local-dir ./Qwen3-14B
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir ./Llama-3.1-8B-Instruct --exclude "original/consolidated*"
```

Flash-attn
```
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
uv pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

```
CUDA_VISIBLE_DEVICES=0 tune run lora_finetune_single_device --config llama3_1_8B_lora_single_device.yaml
```
