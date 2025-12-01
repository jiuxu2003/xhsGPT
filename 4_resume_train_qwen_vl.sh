#!/bin/bash

# 检查是否提供了 checkpoint 路径
if [ -z "$1" ]; then
    echo "错误: 请提供 checkpoint 路径"
    echo "用法: ./4_resume_train_qwen_vl.sh /path/to/checkpoint-xxx"
    exit 1
fi

CHECKPOINT_PATH="$1"
PROJECT_ROOT=$(pwd)
DATA_FILE="$PROJECT_ROOT/qwen_vl_train_data.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/qwen2.5_vl_xhs"
MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"

echo "从 checkpoint 恢复训练: $CHECKPOINT_PATH"

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=0

swift sft \
    --model "$MODEL_ID" \
    --dataset "$DATA_FILE" \
    --train_type lora \
    --target_modules all-linear \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --max_length 4096 \
    --model_kwargs '{"max_pixels": 1003520}' \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4 \
    --report_to swanlab \
    --logging_first_step true \
    --resume_from_checkpoint "$CHECKPOINT_PATH"
