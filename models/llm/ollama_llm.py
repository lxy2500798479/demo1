"""
LLM 对话模块
使用 Ollama 调用本地 Qwen2.5-3B 模型
"""
import asyncio
from typing import Optional, AsyncGenerator, List, Dict
import openai

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT


class OllamaLLM:
    """Ollama LLM 客户端"""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.client = openai.OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",  # Ollama 不需要真正的 key
            timeout=timeout
        )
        self.conversation_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """添加对话历史"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []

    async def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """
        发送聊天请求

        Args:
            message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数
            stream: 是否流式输出

        Returns:
            模型回复
        """
        # 构建消息列表
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.extend(self.conversation_history)
        messages.append({
            "role": "user",
            "content": message
        })

        try:
            if stream:
                # 流式输出
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=True
                )

                full_response = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        # 可以在这里 yield 给前端
                return full_response
            else:
                # 非流式输出
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=False
                )
                return response.choices[0].message.content

        except Exception as e:
            print(f"LLM 调用错误: {e}")
            return f"抱歉，我遇到了错误: {str(e)}"

    async def chat_stream(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        流式聊天

        Args:
            message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数

        Yields:
            模型回复的片段
        """
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.extend(self.conversation_history)
        messages.append({
            "role": "user",
            "content": message
        })

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"LLM 流式调用错误: {e}")
            yield f"抱歉，我遇到了错误: {str(e)}"

    def get_default_system_prompt(self) -> str:
        """获取默认的系统提示词"""
        return """你是一个友好的AI数字人助手，请用简洁、有趣的方式回答用户的问题。
保持对话自然流畅，就像真人在交流一样。"""


# 全局单例
_llm_instance: Optional[OllamaLLM] = None


def get_llm() -> OllamaLLM:
    """获取 LLM 实例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = OllamaLLM()
    return _llm_instance
