"""
LLM 对话模块
使用 vLLM 调用本地 Qwen2.5-3B 模型
"""
import asyncio
from typing import Optional, AsyncGenerator, List, Dict
import openai

from config import VLLM_HOST, VLLM_PORT, VLLM_MODEL, VLLM_API_KEY, VLLM_TIMEOUT


class VLLMClient:
    """vLLM 客户端"""

    def __init__(
        self,
        host: str = VLLM_HOST,
        port: int = VLLM_PORT,
        model: str = VLLM_MODEL,
        api_key: str = VLLM_API_KEY,
        timeout: int = VLLM_TIMEOUT,
    ):
        self.base_url = f"http://{host}:{port}/v1"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=api_key,
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
        """发送聊天请求"""
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
                return full_response
            else:
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
        """流式聊天"""
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
_llm_instance: Optional[VLLMClient] = None


def get_llm() -> VLLMClient:
    """获取 LLM 实例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = VLLMClient()
    return _llm_instance
