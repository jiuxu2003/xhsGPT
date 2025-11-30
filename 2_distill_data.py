import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures

# 配置
INPUT_FILE = 'raw_metadata.json'
OUTPUT_FILE = 'final_train_data.json'
API_KEY = os.getenv("DEEPSEEK_API_KEY") # 这里实际上是百炼的 API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" # 阿里云百炼兼容 OpenAI 接口地址
MAX_WORKERS = 5 # 并发请求数

if not API_KEY:
    print("警告: 未找到 DEEPSEEK_API_KEY 环境变量。请设置后运行。")
    # 为了测试，允许脚本继续，但在调用时会失败
    
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """你是一位小红书爆款文案创作者。你的任务是将输入的英文图片描述转化为一篇极具吸引力的小红书笔记文案。
要求：
1. **语气风格**：热情、亲切、分享欲强，像在和闺蜜聊天。
2. **内容元素**：必须包含 Emoji 表情、语气词（如“哇”、“绝绝子”、“太好看了吧”）、生活感悟。
3. **结构**：标题要吸引眼球，正文简短有力，最后加上 3-5 个相关的 Hashtag。
4. **输入**：一段英文图片描述。
5. **输出**：仅输出中文文案内容，不要包含任何解释性文字。"""

def generate_xhs_caption(english_desc):
    try:
        response = client.chat.completions.create(
            model="deepseek-r1", # 阿里云百炼模型名称
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"图片描述：{english_desc}\n\n请生成小红书文案："}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API 调用失败: {e}")
        return None

def process_item(item):
    image_path = item['image_path']
    raw_caption = item['raw_caption']
    
    # 如果已经处理过（假设有缓存机制，这里简单处理）
    # 在实际生产中可能需要更复杂的断点续传
    
    xhs_caption = generate_xhs_caption(raw_caption)
    
    if xhs_caption:
        return {
            "id": os.path.splitext(os.path.basename(image_path))[0],
            "image": image_path,
            "conversations": [
                {"from": "human", "value": "给这张图配个小红书文案\n<Img><ImageHere></Img>"},
                {"from": "gpt", "value": xhs_caption}
            ]
        }
    return None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"加载了 {len(raw_data)} 条数据。开始蒸馏...")
    
    final_data = []
    
    # 使用线程池并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_item, item) for item in raw_data]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="生成文案"):
            result = future.result()
            if result:
                final_data.append(result)
                
    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
        
    print(f"完成！已保存 {len(final_data)} 条数据到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
