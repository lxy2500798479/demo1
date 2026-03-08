"""
数字人配置中心
"""
import os
from pathlib import Path

# ============== 项目路径 ==============
PROJECT_ROOT = Path(__file__).parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"

# 确保目录存在
ASSETS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============== 服务配置 ==============
HOST = "0.0.0.0"
PORT = 7860
DEBUG = True

# ============== ASR 配置 ==============
WHISPER_MODEL = "tiny"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# ============== LLM 配置 (vLLM) ==============
VLLM_HOST = "0.0.0.0"
VLLM_PORT = 8000
VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # HuggingFace 模型名
VLLM_API_KEY = "EMPTY"
VLLM_TIMEOUT = 120

# ============== TTS 配置 ==============
EDGE_TTS_VOICE = "zh-CN-XiaoxiaoNeural"
EDGE_TTS_RATE = "+0%"
EDGE_TTS_PITCH = "+0Hz"

# ============== MuseTalk 配置 ==============
MUSETALK_DEVICE = "cuda"
MUSETALK_FPS = 25
MUSETALK_BATCH_SIZE = 1

# ============== VAD 配置 ==============
VAD_MODE = 3
VAD_FRAME_DURATION = 30
VAD_SILENCE_THRESHOLD = 500

# ============== WebRTC 配置 ==============
WEBRTC_SAMPLE_RATE = 16000
WEBRTC_CHANNELS = 1
