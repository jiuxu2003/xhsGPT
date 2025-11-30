# 📕 xhsGPT: 小红书风格文案生成助手

> **基于 Qwen2.5-VL-7B 的垂直领域多模态文案生成模型**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--VL--7B-green)](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)
[![Framework](https://img.shields.io/badge/Framework-ms--swift-orange)](https://github.com/modelscope/swift)

## 📖 项目介绍

**xhsGPT** 是一个专门用于生成“小红书风格”文案的多模态 AI 项目。它能够理解图片内容（如风景、美食、穿搭、化妆品），并自动生成包含 Emoji、标签（Hashtags）和独特语气（如“绝绝子”、“家人们”）的高质量文案。

本项目基于 **Qwen2.5-VL-7B-Instruct** 视觉语言模型，使用 **DeepSeek-R1** 蒸馏的高质量数据进行 **LoRA 微调**，在保持模型通用能力的同时，赋予其极强的垂直领域写作能力。

## ✨ 核心功能

*   **👀 看图说话**：精准识别图片细节（OCR、物体检测、场景分析）。
*   **✍️ 风格化写作**：完美复刻小红书博主的语气、排版和表情包使用习惯。
*   **🚀 高效微调**：基于 `ms-swift` 框架，支持在单张 RTX 4090 (24GB) 上进行 LoRA 微调。

## 🛠️ 技术栈

*   **基座模型**: [Qwen2.5-VL-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)
*   **训练框架**: [ms-swift](https://github.com/modelscope/swift)
*   **数据蒸馏**: DeepSeek-R1 (Reasoning Model)
*   **硬件要求**: NVIDIA RTX 4090 (24GB VRAM) 或更高

## 📂 文件结构

```text
.
├── 1_download_unsplash.py    # 数据获取：从 Unsplash 下载高质量图片
├── 2_distill_data.py         # 数据蒸馏：调用 DeepSeek API 生成小红书文案
├── 3_format_qwen_vl.py       # 数据处理：转换为 Qwen2.5-VL 训练格式
├── 4_train_qwen_vl.sh        # 模型训练：一键启动 LoRA 微调脚本
├── 5_inference_qwen_vl.py    # 模型推理：加载微调权重进行测试
└── requirements.txt          # 项目依赖
```

## 🚀 快速开始

### 1. 环境准备

推荐使用 Conda 创建环境：

```bash
conda create -n xhsGPT python=3.10
conda activate xhsGPT

# 安装 PyTorch (CUDA 12.1)
pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple --cache-dir ~/.cache/pip

# 安装项目依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --cache-dir ~/.cache/pip
```

### 2. 数据准备

本项目包含完整的数据处理流：
1.  运行 `1_download_unsplash.py` 下载图片。
2.  运行 `2_distill_data.py` 生成训练数据。
3.  运行 `3_format_qwen_vl.py` 转换为训练格式。

### 3. 模型微调

```bash
chmod +x 4_train_qwen_vl.sh
./4_train_qwen_vl.sh
```

### 4. 推理验证

```bash
python 5_inference_qwen_vl.py
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

Apache 2.0 License
