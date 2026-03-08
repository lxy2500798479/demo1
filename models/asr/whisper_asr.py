"""
语音识别模块 (ASR)
使用 Faster Whisper 进行实时语音识别
"""
import asyncio
import numpy as np
from typing import Optional, AsyncGenerator
import whisper
import torch

from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE


class WhisperASR:
    """Whisper 语音识别器"""

    def __init__(
        self,
        model_name: str = WHISPER_MODEL,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def load_model(self):
        """加载 Whisper 模型"""
        if self.model is None:
            print(f"Loading Whisper model: {self.model_name}...")
            self.model = whisper.load_model(
                self.model_name,
                device=self.device,
                download_root="./models/asr"
            )
            print(f"Whisper model loaded on {self.device}")
        return self.model

    async def recognize(
        self,
        audio_data: np.ndarray,
        language: str = "zh"
    ) -> str:
        """
        识别音频数据

        Args:
            audio_data: numpy 数组，16kHz, mono
            language: 语言代码，zh 为中文

        Returns:
            识别的文本
        """
        if self.model is None:
            self.load_model()

        # Whisper 需要 16kHz 采样率
        # 如果是其他采样率，需要重采样
        result = await asyncio.to_thread(
            self.model.transcribe,
            audio_data,
            language=language,
            fp16=self.device == "cuda"
        )

        return result.get("text", "").strip()

    async def recognize_stream(
        self,
        audio_generator: AsyncGenerator[np.ndarray, None],
        language: str = "zh"
    ) -> AsyncGenerator[str, None]:
        """
        流式识别

        Args:
            audio_generator: 音频数据生成器
            language: 语言代码

        Yields:
            识别的文本片段
        """
        if self.model is None:
            self.load_model()

        buffer = []
        sample_rate = 16000

        async for audio_chunk in audio_generator:
            if audio_chunk is not None and len(audio_chunk) > 0:
                buffer.append(audio_chunk)

                # 每累积一定长度的音频进行一次识别
                if len(buffer) * len(audio_chunk) / sample_rate > 1.0:
                    combined_audio = np.concatenate(buffer)
                    result = await self.recognize(combined_audio, language)
                    if result:
                        yield result
                    buffer = []

        # 处理剩余的音频
        if buffer:
            combined_audio = np.concatenate(buffer)
            result = await self.recognize(combined_audio, language)
            if result:
                yield result

    def unload_model(self):
        """卸载模型，释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# 全局单例
_asr_instance: Optional[WhisperASR] = None


def get_asr() -> WhisperASR:
    """获取 ASR 实例"""
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = WhisperASR()
    return _asr_instance
