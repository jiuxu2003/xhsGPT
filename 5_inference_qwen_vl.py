import os
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

# 配置
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# 训练完成后，将下面的路径替换为 checkpoints 目录下的 latest 文件夹路径
# 例如: "./checkpoints/qwen2.5_vl_xhs/qwen2.5-vl-7b-instruct/v0-20241130-100000/checkpoint-xxx"
CKPT_DIR = None 
TEST_IMAGE = "./dataset/images/example.jpg" # 需替换为实际存在的测试图片

def main():
    # 1. 加载模型和 Tokenizer
    # 如果有微调后的权重，优先加载微调后的权重
    model_path = CKPT_DIR if CKPT_DIR else MODEL_ID
    
    print(f"Loading model from {model_path}...")
    model, tokenizer = get_model_tokenizer(model_path, model_kwargs={'device_map': 'auto'})
    
    # 2. 获取模板
    template_type = get_default_template_type(model_path)
    template = get_template(template_type, tokenizer)
    
    # 3. 构造输入
    # Qwen2-VL 推荐使用 <image> 占位符
    query = "<image>\n请为这张图片写一段小红书文案"
    
    # 4. 推理
    print("Generating...")
    response, history = inference(
        model, 
        template, 
        query, 
        images=[TEST_IMAGE]
    )
    
    print("-" * 20)
    print("Generated Output:")
    print(response)
    print("-" * 20)

if __name__ == "__main__":
    # 简单的检查
    if not os.path.exists(TEST_IMAGE):
        # 尝试找一张存在的图片
        import glob
        images = glob.glob("./dataset/images/*.jpg")
        if images:
            TEST_IMAGE = images[0]
            print(f"Found image: {TEST_IMAGE}")
        else:
            print(f"Warning: No images found in ./dataset/images/. Please check.")
            
    if os.path.exists(TEST_IMAGE):
        main()
