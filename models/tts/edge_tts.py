"""
TTS 语音合成模块
使用 Microsoft Edge TTS 进行语音合成
"""
import asyncio
import os
import tempfile
from typing import Optional, AsyncGenerator
import edge_tts
import numpy as np
import soundfile as sf

from config import EDGE_TTS_VOICE, EDGE_TTS_RATE, EDGE_TTS_PITCH


class EdgeTTS:
    """Edge TTS 语音合成器"""

    def __init__(
        self,
        voice: str = EDGE_TTS_VOICE,
        rate: str = EDGE_TTS_RATE,
        pitch: str = EDGE_TTS_PITCH,
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    async def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        合成语音

        Args:
            text: 要合成的中文
            output_path: 输出文件路径，如果为 None 则创建临时文件

        Returns:
            生成的音频文件路径
        """
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp3")

        communicate = edge_tts.Communicate(
            text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )

        await communicate.save(output_path)
        return output_path

    async def synthesize_stream(
        self,
        text: str,
    ) -> AsyncGenerator[bytes, None]:
        """
        流式合成语音（边说边返回）

        Args:
            text: 要合成的文本

        Yields:
            音频数据块
        """
        communicate = edge_tts.Communicate(
            text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def synthesize_and_load(
        self,
        text: str,
    ) -> tuple[np.ndarray, int]:
        """
        合成语音并直接加载为 numpy 数组

        Args:
            text: 要合成的文本

        Returns:
            (音频数据, 采样率)
        """
        # 创建临时文件
        temp_file = tempfile.mktemp(suffix=".mp3")

        try:
            # 合成
            communicate = edge_tts.Communicate(
                text,
                voice=self.voice,
                rate=self.rate,
                pitch=self.pitch
            )
            await communicate.save(temp_file)

            # 加载为 numpy
            audio, sample_rate = sf.read(temp_file)

            # 转换为 float32 如果需要
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # 如果是立体声，转为单声道
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            return audio, sample_rate

        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def get_available_voices(self) -> list[str]:
        """获取可用的语音列表"""
        # 常用的中文语音
        return [
            "zh-CN-XiaoxiaoNeural",      # 晓晓 - 女声
            "zh-CN-YunxiNeural",        # 云希 - 男声
            "zh-CN-YunyangNeural",      # 云扬 - 男声
            "zh-CN-XiaoyiNeural",       # 晓伊 - 女声
            "zh-CN-YunfengNeural",      # 云风 - 男声
        ]

    async def get_audio_duration(self, text: str) -> float:
        """估算音频时长（秒）"""
        # 粗略估算：中文每字约 0.25 秒
        return len(text) * 0.25


# 全局单例
_tts_instance: Optional[EdgeTTS] = None


def get_tts() -> EdgeTTS:
    """获取 TTS 实例"""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = EdgeTTS()
    return _tts_instance
