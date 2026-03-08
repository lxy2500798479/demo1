"""
MuseTalk 口型驱动模块
用于根据音频驱动数字人形象的口型
"""
import os
import asyncio
from pathlib import Path
from typing import Optional, List, AsyncGenerator
import numpy as np
import torch
import cv2
from PIL import Image

from config import MUSETALK_DEVICE, MUSETALK_FPS, MUSETALK_BATCH_SIZE


class MuseTalkDriver:
    """MuseTalk 数字人口型驱动"""

    def __init__(
        self,
        device: str = MUSETALK_DEVICE,
        fps: int = MUSETALK_FPS,
    ):
        self.device = device
        self.fps = fps
        self.model = None
        self.face_analysis = None
        self.reference_image: Optional[Image.Image] = None

    def _download_models(self):
        """下载 MuseTalk 模型"""
        # MuseTalk 模型路径
        model_dir = Path("./models/talk/musetalk")
        model_dir.mkdir(parents=True, exist_ok=True)

        # 检查模型是否存在
        needed_files = [
            "musetalk.json",
            "pytorch_model.bin",
            "unet.pt",
            "faceparser.pt",
            "semantic_encoder",
        ]

        # 如果模型不存在，打印下载说明
        missing = [f for f in needed_files if not (model_dir / f).exists()]
        if missing:
            print("MuseTalk 模型未找到，请手动下载:")
            print("https://github.com/TMElyralab/MuseTalk")

        return model_dir

    def load_model(self):
        """加载 MuseTalk 模型"""
        if self.model is not None:
            return

        print("Loading MuseTalk model...")

        try:
            # 尝试导入 MuseTalk
            # 注意: 需要先安装 museTalk
            # pip install museTalk
            from musetalk.model import MuseTalk
            from musetalk.utils.utils import get_video_data
            from musetalk.utils.blink import Blink

            model_dir = self._download_models()

            # 加载模型
            self.model = MuseTalk(
                model_dir=str(model_dir),
                device=self.device
            )

            print(f"MuseTalk model loaded on {self.device}")

        except ImportError:
            print("MuseTalk 未安装，请运行: pip install museTalk")
            raise

    def set_reference_image(self, image_path: str):
        """
        设置参考图像（数字人形象）

        Args:
            image_path: 参考图片路径
        """
        self.reference_image = Image.open(image_path).convert("RGB")

        # 调整图像大小为 256x256
        self.reference_image = self.reference_image.resize((256, 256))

        if self.model is None:
            self.load_model()

    async def generate(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        根据音频生成说话视频

        Args:
            audio_path: 音频文件路径
            output_path: 输出视频路径

        Returns:
            生成的视频路径
        """
        if self.reference_image is None:
            raise ValueError("请先设置参考图像 (set_reference_image)")

        if self.model is None:
            self.load_model()

        if output_path is None:
            output_path = "output.mp4"

        # 转换参考图像为模型输入格式
        # ... (MuseTalk 具体调用)

        print(f"Generating video from audio: {audio_path}")
        return output_path

    async def generate_stream(
        self,
        audio_chunks: List[str],
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        流式生成视频帧

        Args:
            audio_chunks: 音频块列表

        Yields:
            视频帧 (numpy array)
        """
        if self.reference_image is None:
            raise ValueError("请先设置参考图像")

        # 实现流式生成逻辑
        # ... 
        pass

    def unload_model(self):
        """卸载模型，释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class SimpleAvatarDriver:
    """
    简化版数字人驱动（不依赖 MuseTalk）
    用于快速测试和开发
    """

    def __init__(
        self,
        fps: int = 25,
    ):
        self.fps = fps
        self.reference_image: Optional[np.ndarray] = None

    def set_reference_image(self, image_path: str):
        """设置参考图像"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        self.reference_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def generate_video(
        self,
        audio_path: str,
        output_path: str,
    ) -> str:
        """
        生成视频（简化版：只输出静态图 + 音频）

        Args:
            audio_path: 音频路径
            output_path: 输出视频路径

        Returns:
            输出视频路径
        """
        if self.reference_image is None:
            raise ValueError("请先设置参考图像")

        import subprocess

        # 使用 ffmpeg 将音频和图像合成为视频
        # 简化版：直接复制图像作为视频
        cmd = [
            "ffmpeg",
            "-loop", "1",
            "-i", self.reference_image_to_temp(),
            "-i", audio_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg 错误: {e}")
            # 备选方案：只返回音频
            return audio_path

        return output_path

    def reference_image_to_temp(self) -> str:
        """保存参考图像到临时文件"""
        temp_path = "/tmp/avatar_ref.jpg"
        if self.reference_image is not None:
            cv2.imwrite(temp_path, cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2BGR))
        return temp_path


# 全局单例
_musetalk_instance: Optional[MuseTalkDriver] = None
_simple_driver_instance: Optional[SimpleAvatarDriver] = None


def get_musetalk(use_simple: bool = True) -> Optional[MuseTalkDriver]:
    """获取 MuseTalk 实例"""
    global _musetalk_instance
    if use_simple:
        return None  # 使用简化版
    if _musetalk_instance is None:
        _musetalk_instance = MuseTalkDriver()
    return _musetalk_instance


def get_simple_driver() -> SimpleAvatarDriver:
    """获取简化版驱动实例"""
    global _simple_driver_instance
    if _simple_driver_instance is None:
        _simple_driver_instance = SimpleAvatarDriver()
    return _simple_driver_instance
