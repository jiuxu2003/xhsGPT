import pandas as pd
import requests
import os
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# 配置
TSV_FILE = sys.argv[1] if len(sys.argv) > 1 else 'photos.csv000'
OUTPUT_DIR = './dataset/images/'
METADATA_FILE = 'raw_metadata.json'
SAMPLE_SIZE = 5000
MAX_WORKERS = 10  # 下载线程数
TIMEOUT = 10      # 请求超时时间（秒）
MAX_RETRIES = 3   # 失败重试次数

def download_image(row):
    """
    下载单张图片，成功则返回元数据。
    """
    photo_id = row['photo_id']
    photo_url = row['photo_image_url']
    description = row['photo_description'] if pd.notna(row['photo_description']) else ""
    ai_description = row['ai_description'] if pd.notna(row['ai_description']) else ""
    
    # 优先使用用户描述，其次使用 AI 描述
    caption = description if description else ai_description
    
    if not caption:
        return None # 如果没有描述则跳过

    image_filename = f"{photo_id}.jpg"
    image_path = os.path.join(OUTPUT_DIR, image_filename)
    temp_path = image_path + ".tmp"

    # 检查是否已存在（仅当完整文件存在时才跳过）
    if os.path.exists(image_path):
        return {
            "image_path": image_path,
            "raw_caption": caption
        }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(photo_url, timeout=TIMEOUT, stream=True)
            if response.status_code == 200:
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                # 下载完成后重命名，确保原子性
                os.rename(temp_path, image_path)
                return {
                    "image_path": image_path,
                    "raw_caption": caption
                }
            elif response.status_code == 404:
                return None # 图片未找到，跳过
            else:
                time.sleep(1) # 重试前等待
        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            time.sleep(1) # 重试前等待
            
    return None # 重试后失败

def main():
    # 1. 设置目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建目录: {OUTPUT_DIR}")

    # 2. 读取 TSV 文件
    print(f"正在读取 {TSV_FILE}...")
    try:
        # 假设标准 Unsplash Lite TSV 格式
        df = pd.read_csv(TSV_FILE, sep='\t', on_bad_lines='skip')
    except FileNotFoundError:
        print(f"错误: 找不到文件 {TSV_FILE}。请确保文件存在。")
        return
    except Exception as e:
        print(f"读取 TSV 文件时出错: {e}")
        return

    print(f"TSV 文件总行数: {len(df)}")

    # 3. 过滤和采样
    # 确保所需列存在
    required_cols = ['photo_id', 'photo_image_url']
    if not all(col in df.columns for col in required_cols):
        print(f"错误: TSV 中缺少必要的列。发现的列: {df.columns}")
        return

    # 过滤包含描述或 AI 描述的行
    has_desc = 'photo_description' in df.columns
    has_ai_desc = 'ai_description' in df.columns
    
    if not has_desc and not has_ai_desc:
        print("错误: 未找到描述列。")
        return

    mask = pd.Series([False] * len(df))
    if has_desc:
        mask |= df['photo_description'].notna()
    if has_ai_desc:
        mask |= df['ai_description'].notna()
        
    filtered_df = df[mask]
    print(f"包含描述的行数: {len(filtered_df)}")

    if len(filtered_df) == 0:
        print("未找到有效数据。")
        return

    # 采样
    sample_n = min(SAMPLE_SIZE, len(filtered_df))
    sampled_df = filtered_df.sample(n=sample_n, random_state=42)
    print(f"采样了 {sample_n} 张图片进行下载。")

    # 4. 下载图片
    results = []
    print(f"开始下载，使用 {MAX_WORKERS} 个线程...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = [executor.submit(download_image, row) for _, row in sampled_df.iterrows()]
        
        # 使用进度条处理结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="下载中"):
            res = future.result()
            if res:
                results.append(res)

    print(f"成功下载 {len(results)} 张图片。")

    # 5. 保存元数据
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"元数据已保存至 {METADATA_FILE}")

if __name__ == "__main__":
    main()
