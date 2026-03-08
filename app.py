"""
数字人主程序
实时语音对话 + 口型同步
"""
import asyncio
import base64
import io
import json
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional

import gradio as gr
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
from PIL import Image

from config import HOST, PORT, DEBUG, PROJECT_ROOT, ASSETS_DIR
from models.asr.whisper_asr import WhisperASR, get_asr
from models.llm.vllm_llm import VLLMClient, get_llm
from models.tts.edge_tts import EdgeTTS, get_tts
from models.talk.musetalk_driver import get_simple_driver, SimpleAvatarDriver


# ============== 全局状态 ==============
avatar_driver: Optional[SimpleAvatarDriver] = None
current_avatar_path: Optional[str] = None
asr_model: Optional[WhisperASR] = None
llm_model: Optional[VLLMClient] = None
tts_model: Optional[EdgeTTS] = None


# ============== 初始化 ==============
def initialize_models():
    """初始化所有模型"""
    global asr_model, llm_model, tts_model, avatar_driver

    print("初始化模型...")

    # ASR
    try:
        asr_model = get_asr()
        print("✓ ASR 模型已加载")
    except Exception as e:
        print(f"✗ ASR 模型加载失败: {e}")

    # LLM
    try:
        llm_model = get_llm()
        print("✓ LLM 模型已加载")
    except Exception as e:
        print(f"✗ LLM 模型加载失败: {e}")

    # TTS
    try:
        tts_model = get_tts()
        print("✓ TTS 模型已加载")
    except Exception as e:
        print(f"✗ TTS 模型加载失败: {e}")

    # 口型驱动
    try:
        avatar_driver = get_simple_driver()
        print("✓ 口型驱动已加载")
    except Exception as e:
        print(f"✗ 口型驱动加载失败: {e}")


# ============== 核心处理逻辑 ==============
async def process_voice_input(audio_data: bytes) -> str:
    """
    处理语音输入

    Args:
        audio_data: 音频字节数据

    Returns:
        识别的文本
    """
    if asr_model is None:
        return "ASR 模型未初始化"

    try:
        # 将 bytes 转换为 numpy 数组
        # 这里假设音频是 16kHz, 16bit, mono
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0

        # 识别
        text = await asr_model.recognize(audio_array)
        return text

    except Exception as e:
        print(f"语音识别错误: {e}")
        return ""


async def get_llm_response(text: str) -> str:
    """
    获取 LLM 回答

    Args:
        text: 用户输入

    Returns:
        LLM 回复
    """
    if llm_model is None:
        return "LLM 模型未初始化"

    try:
        response = await llm_model.chat(text)
        return response

    except Exception as e:
        print(f"LLM 错误: {e}")
        return f"抱歉，我遇到了错误: {str(e)}"


async def synthesize_speech(text: str) -> tuple[str, float]:
    """
    合成语音

    Args:
        text: 要合成的文本

    Returns:
        (音频路径, 时长)
    """
    if tts_model is None:
        return "", 0.0

    try:
        audio_path = await tts_model.synthesize(text)
        return audio_path, 0.0  # TODO: 获取实际时长

    except Exception as e:
        print(f"TTS 错误: {e}")
        return "", 0.0


# ============== Gradio 前端 ==============
def create_demo():
    """创建 Gradio 界面"""

    with gr.Blocks(title="数字人助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎭 实时数字人助手")
        gr.Markdown("上传你的照片，开始实时语音对话吧！")

        with gr.Row():
            with gr.Column(scale=1):
                # 头像上传
                avatar_input = gr.Image(
                    label="上传数字人形象",
                    type="filepath",
                    height=300,
                )
                avatar_submit = gr.Button("设置形象", variant="primary")

                # 状态显示
                status_text = gr.Textbox(
                    label="状态",
                    value="等待上传形象...",
                    interactive=False,
                )

                # 语音输入
                audio_input = gr.Audio(
                    label="按住说话",
                    type="numpy",
                    sources=["microphone"],
                )

            with gr.Column(scale=2):
                # 视频输出区域
                video_output = gr.Image(
                    label="数字人视频",
                    height=400,
                )

                # 文本对话
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=200,
                )
                text_input = gr.Textbox(
                    label="输入文本",
                    placeholder="也可以直接输入文字...",
                )
                text_submit = gr.Button("发送", variant="primary")

        # 事件处理
        def set_avatar(image_path):
            global current_avatar_path, avatar_driver
            if image_path is None:
                return "请上传图片"

            current_avatar_path = image_path
            if avatar_driver is not None:
                avatar_driver.set_reference_image(image_path)

            return f"形象已设置: {image_path}"

        async def process_audio(audio):
            if audio is None:
                return None, "请先录音"

            # 提取音频数据
            sample_rate, audio_data = audio

            # 识别语音
            text = await process_voice_input(audio_data.tobytes())

            if not text:
                return None, "未识别到语音"

            # 获取回复
            response = await get_llm_response(text)

            # 合成语音
            audio_path, _ = await synthesize_speech(response)

            # 更新视频
            video_frame = current_avatar_path  # 简化版：显示原图

            return video_frame, f"你说: {text}\n我回复: {response}"

        async def process_text(text, history):
            if not text:
                return "", history

            # 获取回复
            response = await get_llm_response(text)

            # 合成语音
            audio_path, _ = await synthesize_speech(response)

            # 更新对话历史
            history.append((text, response))

            return "", history

        # 绑定事件
        avatar_submit.click(
            set_avatar,
            inputs=[avatar_input],
            outputs=[status_text],
        )

        audio_input.change(
            process_audio,
            inputs=[audio_input],
            outputs=[video_output, chatbot],
        )

        text_submit.click(
            process_text,
            inputs=[text_input, chatbot],
            outputs=[text_input, chatbot],
        )

        text_input.submit(
            process_text,
            inputs=[text_input, chatbot],
            outputs=[text_input, chatbot],
        )

    return demo


# ============== FastAPI 应用 ==============
app = FastAPI(title="数字人 API")


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    initialize_models()


@app.get("/")
async def root():
    """根路径"""
    return {"message": "数字人 API 服务", "status": "running"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 实时对话"""
    await websocket.accept()

    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()

            message_type = data.get("type")

            if message_type == "audio":
                # 处理音频
                audio_base64 = data.get("data")
                audio_bytes = base64.b64decode(audio_base64)

                # 识别
                text = await process_voice_input(audio_bytes)

                if text:
                    # 获取回复
                    response = await get_llm_response(text)

                    # 合成语音
                    audio_path, _ = await synthesize_speech(response)

                    # 读取音频文件并转为 base64
                    with open(audio_path, "rb") as f:
                        audio_data = base64.b64encode(f.read()).decode()

                    # 返回
                    await websocket.send_json({
                        "type": "response",
                        "text": text,
                        "response": response,
                        "audio": audio_data,
                    })

            elif message_type == "text":
                # 处理文本
                text = data.get("text")
                response = await get_llm_response(text)

                # 合成语音
                audio_path, _ = await synthesize_speech(response)

                # 读取音频
                with open(audio_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode()

                await websocket.send_json({
                    "type": "response",
                    "text": text,
                    "response": response,
                    "audio": audio_data,
                })

    except Exception as e:
        print(f"WebSocket 错误: {e}")
    finally:
        await websocket.close()


# ============== 主入口 ==============
if __name__ == "__main__":
    # 初始化
    initialize_models()

    # 启动 Gradio
    print(f"\n🚀 启动数字人服务: http://localhost:{PORT}")
    demo = create_demo()
    demo.launch(
        server_name=HOST,
        server_port=PORT,
        share=False,
        inbrowser=DEBUG,
    )
