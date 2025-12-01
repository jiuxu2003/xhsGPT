import os
import torch
import argparse
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# 配置
# 配置
# 使用本地模型路径 (ms-swift 默认下载路径)
MODEL_ID = "/data/models/models/Qwen/Qwen2.5-VL-7B-Instruct"
# 如果本地路径不存在，尝试使用 ModelScope ID
if not os.path.exists(MODEL_ID):
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# 默认 Checkpoint 路径 (会被命令行参数覆盖)
DEFAULT_CKPT_DIR = "/data/xhsGPT/checkpoints/qwen2.5_vl_xhs/v4-20251130-163010/checkpoint-939"

def main():
    parser = argparse.ArgumentParser(description='Qwen2.5-VL Inference (Transformers)')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--base', action='store_true', help='Use base model only (ignore adapter)')
    parser.add_argument('--ckpt', type=str, default=DEFAULT_CKPT_DIR, help='Path to LoRA checkpoint')
    args = parser.parse_args()

    # 1. 检查图片
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        exit(1)
    
    image_path = os.path.abspath(args.image)
    print(f"Processing image: {image_path}")

    # 2. 加载模型
    print(f"Loading base model: {MODEL_ID}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2", # 既然没有安装 flash_attn，就让它自动选择 (通常是 sdpa)
        device_map="auto",
    )
    
    # 3. 加载 Processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 4. 加载 LoRA (如果不使用 --base)
    if not args.base:
        if args.ckpt and os.path.exists(args.ckpt):
            print(f"Loading LoRA adapter from: {args.ckpt}")
            model = PeftModel.from_pretrained(model, args.ckpt)
        else:
            print(f"Warning: Checkpoint path {args.ckpt} not found. Using base model.")
    else:
        print("Using base model only (ignoring adapter).")

    # 5. 构造输入
    # Qwen2.5-VL 的标准输入格式
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": "请仔细观察这张图片，描述图片中的内容（场景、物体、氛围），然后基于这些内容写一段吸引人的小红书文案。文案要包含emoji和标签。"},
            ],
        }
    ]

    # 6. 预处理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 7. 推理
    print("Generating...")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("-" * 20)
    print("Generated Output:")
    print(output_text[0])
    print("-" * 20)

if __name__ == "__main__":
    main()
