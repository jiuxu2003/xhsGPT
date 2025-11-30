#!/bin/bash

# Qwen2-VL-7B Fine-tuning Script using ms-swift
# 适配 RTX 4090 (24GB VRAM)

# 1. 基础配置
PROJECT_ROOT=$(pwd)
DATA_FILE="$PROJECT_ROOT/qwen_vl_train_data.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/qwen2.5_vl_xhs"
MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"

# 2. 启动训练
# ms-swift 会自动处理模型下载和加载
# 使用 LoRA 微调

echo "开始训练 Qwen2-VL-7B..."
echo "数据文件: $DATA_FILE"

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=0

# swift sft 参数说明:
# --model_type: 指定模型类型 (swift 会自动识别)
# --dataset: 数据集路径
# --sft_type: lora
# --target_modules: 指定 LoRA 目标模块 (ALL 表示所有线性层)
# --batch_size: 显存优化，设为 1
# --gradient_accumulation_steps: 梯度累积，设为 16 (相当于 batch_size 16)
# --learning_rate: 学习率
# --num_train_epochs: 训练轮数
# --max_length: 序列最大长度 (Qwen2-VL 支持长序列，但为了显存设为 2048 或更小)

swift sft \
    --model_id_or_path "$MODEL_ID" \
    --dataset "$DATA_FILE" \
    --sft_type lora \
    --target_modules ALL \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --max_length 2048 \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4

echo "训练完成！模型保存在: $OUTPUT_DIR"
