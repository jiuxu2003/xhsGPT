# Qwen2.5-VL-7B 小红书文案生成模型训练指南

## 1. 项目概述
本项目使用 **Qwen2.5-VL-7B-Instruct** 模型，通过 **ms-swift** 框架进行 **LoRA 微调**，使其具备生成“小红书风格”文案的能力。

## 2. 环境准备 (服务器端)

### 2.1 安装依赖
建议使用 Conda 创建环境 (Python 3.10+)：
```bash
conda create -n xhsGPT python=3.10
conda activate xhsGPT

# 1. 先安装 PyTorch (CUDA 12.1 版本)
pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple --cache-dir ~/.cache/pip

# 2. 再安装其他依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --cache-dir ~/.cache/pip
```

### 2.2 准备数据
**重要**：`unsplash-lite.zip` 只是元数据，**不要**只上传它。
请务必将本地已经跑通的以下文件/文件夹上传到服务器：
1.  `dataset/` 文件夹 (里面包含了 5000 张已下载的图片)
2.  `final_train_data.json` (本地蒸馏好的数据)

确保它们位于项目根目录下，然后运行转换脚本：
```bash
python 3_format_qwen_vl.py
```
这将生成 `qwen_vl_train_data.jsonl`。

### 2.3 配置 SwanLab (可视化监控)
本项目已集成 SwanLab 用于监控 Loss 曲线。
1.  注册 [SwanLab](https://swanlab.cn) 账号并获取 API Key。
2.  在服务器终端运行：
    ```bash
    swanlab login
    # 粘贴您的 API Key 并回车
    ```
    或者直接设置环境变量：
    ```bash
    export SWANLAB_API_KEY="your_api_key_here"
    ```

## 3. 开始训练

运行训练脚本：
```bash
chmod +x 4_train_qwen_vl.sh
./4_train_qwen_vl.sh
```

**说明**:
- 脚本会自动从 ModelScope 下载 Qwen2.5-VL-7B-Instruct 模型。
- 训练过程中会使用 Flash Attention 2 加速（如果硬件支持）。
- 显存占用预计在 16GB - 20GB 之间 (RTX 4090 可用)。

## 4. 推理验证

训练完成后，修改 `5_inference_qwen_vl.py` 中的 `CKPT_DIR` 为实际生成的 checkpoint 路径（在 `checkpoints/qwen_vl_xhs/` 下）。

运行推理：
```bash
python 5_inference_qwen_vl.py
```

## 5. 常见问题
- **OOM (显存不足)**: 请在 `4_train_qwen_vl.sh` 中将 `max_length` 调小 (如 1024)，或减少 `batch_size` (虽然已经是 1 了)。
- **网络问题**: 如果无法下载模型，请设置 ModelScope 镜像或手动下载模型权重。
