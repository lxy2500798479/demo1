# Realtime Digital Human Avatar
# 实时语音对话数字人 (MuseTalk方案)

## 功能特性
- 实时语音对话 (WebRTC)
- 自定义数字人形象 (上传照片)
- 口型同步 (MuseTalk)
- 本地LLM推理 (Qwen2.5-3B)

## 环境要求
- Python 3.10+
- CUDA 12.1+ (GPU推理)
- 24GB+ 显存 (推荐)

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载模型
```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install | sh

# 下载 Qwen2.5-3B
ollama pull qwen2.5:3b

# MuseTalk 模型会自动下载
```

### 3. 启动服务
```bash
python app.py
```

### 4. 访问
浏览器打开 http://localhost:7860
