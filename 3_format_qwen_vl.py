import json
import os

INPUT_FILE = 'final_train_data.json'
OUTPUT_FILE = 'qwen_vl_train_data.jsonl'

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in data:
            # PandaGPT format:
            # {
            #   "id": "...",
            #   "image": "path/to/image",
            #   "conversations": [
            #       {"from": "human", "value": "... <Img><ImageHere></Img>"},
            #       {"from": "gpt", "value": "..."}
            #   ]
            # }
            
            image_path = item['image']
            # Ensure absolute path if necessary. Assuming relative to project root.
            
            conversations = item['conversations']
            human_input = conversations[0]['value']
            gpt_output = conversations[1]['value']
            
            # Clean instruction
            # Qwen2-VL in ms-swift expects <image> tag to denote image position
            instruction = human_input.replace('<Img><ImageHere></Img>', '').strip()
            instruction = f"<image>\n{instruction}"
            
            # ms-swift Multimodal JSONL Format
            entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": instruction
                    },
                    {
                        "role": "assistant",
                        "content": gpt_output
                    }
                ],
                "images": [image_path]
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Converted {len(data)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
