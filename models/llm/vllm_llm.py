"""
LLM 对话模块
使用 vLLM 调用本地 Qwen2.5-3B 模型
"""
import asyncio
import time
from typing import Optional, AsyncGenerator, List, Dict
import openai
import requests
import urllib3
# 禁用 SSL 警告（如果需要）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import VLLM_HOST, VLLM_PORT, VLLM_MODEL, VLLM_API_KEY, VLLM_TIMEOUT


def check_vllm_health() -> dict:
    """检查 vLLM 服务健康状态"""
    try:
        base_url = f"http://{VLLM_HOST}:{VLLM_PORT}"
        
        print(f"[VLLM] 健康检查 - 测试连接: {base_url}")
        
        # 检查 /v1/models
        models_response = requests.get(f"{base_url}/v1/models", timeout=5)
        models_status = models_response.status_code
        models_data = models_response.json() if models_response.status_code == 200 else None
        
        # 检查 /health
        health_response = requests.get(f"{base_url}/health", timeout=5)
        health_status = health_response.status_code
        
        # 检查 /ping
        ping_response = requests.get(f"{base_url}/ping", timeout=5)
        ping_status = ping_response.status_code
        
        # 测试 chat 接口
        test_chat_response = None
        try:
            chat_response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": VLLM_MODEL,
                    "messages": [{"role": "user", "content": "你好"}],
                    "max_tokens": 50
                },
                timeout=30
            )
            test_chat_status = chat_response.status_code
            test_chat_response = chat_response.text[:200] if chat_response.status_code != 200 else None
        except Exception as e:
            test_chat_status = str(e)
        
        return {
            "connected": True,
            "v1_models_status": models_status,
            "health_status": health_status,
            "ping_status": ping_status,
            "test_chat_status": test_chat_status,
            "test_chat_response": test_chat_response,
            "models_data": models_data,
            "base_url": base_url
        }
    except Exception as e:
        import traceback
        return {
            "connected": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


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
        
        # 打印初始化信息
        print(f"[VLLM] 初始化 vLLM 客户端")
        print(f"[VLLM] base_url: {self.base_url}")
        print(f"[VLLM] model: {self.model}")
        print(f"[VLLM] api_key: {self.api_key}")
        print(f"[VLLM] timeout: {self.timeout}")

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

        # 打印详细的请求信息
        print(f"\n{'='*60}")
        print(f"[VLLM] 请求开始")
        print(f"[VLLM] URL: {self.base_url}/chat/completions")
        print(f"[VLLM] Model: {self.model}")
        print(f"[VLLM] Messages: {messages}")
        print(f"[VLLM] Temperature: {temperature}")
        print(f"{'='*60}\n")

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
                start_time = time.time()
                print(f"[VLLM] 正在调用 vLLM 服务...")
                
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=False
                )
                
                elapsed_time = time.time() - start_time
                print(f"[VLLM] vLLM 响应耗时: {elapsed_time:.2f}秒")
                print(f"[VLLM] 响应内容: {response}")
                print(f"[VLLM] 响应 choices: {response.choices}")
                
                result = response.choices[0].message.content
                print(f"[VLLM] 返回结果: {result[:200]}...")
                print(f"{'='*60}\n")
                
                return result

        except openai.APIConnectionError as e:
            print(f"[VLLM] 连接错误: {e}")
            print(f"[VLLM] 请确保 vLLM 服务已启动在: http://{VLLM_HOST}:{VLLM_PORT}")
            print(f"[VLLM] 启动 vLLM 服务的命令示例:")
            print(f"[VLLM]   vllm serve {VLLM_MODEL} --host {VLLM_HOST} --port {VLLM_PORT} --api-key {VLLM_API_KEY}")
            return f"抱歉，我遇到了错误: 无法连接到 vLLM 服务 - {str(e)}"
        except openai.APIStatusError as e:
            print(f"[VLLM] API 状态错误: {e.status_code} - {e.response}")
            print(f"[VLLM] 响应内容: {e.response.text if e.response else 'N/A'}")
            return f"抱歉，我遇到了错误: Error code: {e.status_code}"
        except Exception as e:
            print(f"[VLLM] 未知错误: {type(e).__name__}: {e}")
            import traceback
            print(f"[VLLM] 详细堆栈: {traceback.format_exc()}")
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
