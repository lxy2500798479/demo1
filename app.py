"""
数字人后端 API 服务
文字输入 -> LLM -> TTS -> 返回音频
"""
import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from config import HOST, PORT, PROJECT_ROOT
from models.llm.vllm_llm import get_llm
from models.tts.edge_tts import get_tts

# 全局模型
llm_model = None
tts_model = None

# 临时文件目录
TEMP_DIR = PROJECT_ROOT / "temp_outputs"
TEMP_DIR.mkdir(exist_ok=True)


class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    text: str
    response: str
    audio_file: str = None


def initialize_models():
    """初始化模型"""
    global llm_model, tts_model

    print("初始化模型...")

    # LLM
    try:
        llm_model = get_llm()
        print("✓ LLM 模型已加载")
    except Exception as e:
        print(f"✗ LLM 模型加载失败: {e}")
        raise

    # TTS - Edge TTS 在线
    try:
        tts_model = get_tts()
        print("✓ TTS 模型已加载")
    except Exception as e:
        print(f"✗ TTS 模型加载失败: {e}")
        raise

    print("模型初始化完成")


app = FastAPI(title="数字人 API")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    from models.llm.vllm_llm import check_vllm_health
    
    vllm_status = check_vllm_health()
    
    return {
        "status": "ok" if vllm_status.get("connected") else "error",
        "vllm": vllm_status
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端页面"""
    html_path = PROJECT_ROOT / "web" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(content=get_default_html())


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理对话请求"""
    print(f"\n{'='*60}")
    print(f"[API] 收到聊天请求")
    print(f"[API] 输入文本: {request.text}")
    print(f"{'='*60}")
    
    if not request.text:
        raise HTTPException(status_code=400, text="输入不能为空")

    if llm_model is None:
        raise HTTPException(status_code=500, text="LLM 模型未初始化")

    try:
        print(f"[API] 步骤1: 调用 LLM 获取回复...")
        
        # 1. 调用 LLM 获取回复
        response_text = await llm_model.chat(request.text)
        
        print(f"[API] LLM 返回: {response_text[:200] if response_text else 'None'}...")

        # 2. 调用 TTS 合成语音
        audio_path = None
        audio_filename = None
        if tts_model:
            print(f"[API] 步骤2: 调用 TTS 合成语音...")
            # 生成唯一文件名
            import uuid
            audio_filename = f"{uuid.uuid4()}.mp3"
            audio_path = str(TEMP_DIR / audio_filename)
            await tts_model.synthesize(response_text, output_path=audio_path)
            print(f"[API] TTS 生成的音频文件: {audio_filename}")

        # 3. 返回结果
        result = ChatResponse(
            text=request.text,
            response=response_text,
            audio_file=audio_filename
        )
        
        print(f"[API] 请求处理完成")
        print(f"{'='*60}\n")
        
        return result

    except Exception as e:
        print(f"[API] 处理错误: {type(e).__name__}: {e}")
        import traceback
        print(f"[API] 详细堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, text=str(e))


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """获取音频文件"""
    audio_path = TEMP_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, text="音频文件不存在")
    return FileResponse(audio_path, media_type="audio/mpeg")


def get_default_html():
    """默认的简单前端页面"""
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数字人对话</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            width: 100%;
            max-width: 600px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        
        h1 {
            text-align: center;
            color: #1a1a2e;
            margin-bottom: 30px;
            font-size: 24px;
        }
        
        .chat-box {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
            background: #fafafa;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 85%;
            line-height: 1.5;
        }
        
        .user-message {
            background: #007AFF;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        
        .assistant-message {
            background: #E9E9EB;
            color: #000;
            border-bottom-left-radius: 4px;
        }
        
        .input-area {
            display: flex;
            gap: 10px;
        }
        
        input[type="text"] {
            flex: 1;
            padding: 14px 18px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus {
            border-color: #007AFF;
        }
        
        button {
            padding: 14px 28px;
            background: #007AFF;
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        button:hover {
            background: #0056b3;
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .loading {
            text-align: center;
            color: #666;
            padding: 20px;
        }
        
        .audio-player {
            margin-top: 10px;
        }
        
        .audio-player audio {
            width: 100%;
            height: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>💬 对话</h1>
        
        <div class="chat-box" id="chatBox">
            <div class="message assistant-message">
                你好！有什么我可以帮你的吗？
            </div>
        </div>
        
        <div class="input-area">
            <input type="text" id="userInput" placeholder="输入文字..." 
                   onkeypress="if(event.key==='Enter')sendMessage()">
            <button id="sendBtn" onclick="sendMessage()">发送</button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chatBox');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        
        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;
            
            // 显示用户消息
            addMessage(text, 'user');
            userInput.value = '';
            
            // 显示加载状态
            sendBtn.disabled = true;
            sendBtn.textContent = '处理中...';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text})
                });
                
                if (!response.ok) {
                    throw new Error('请求失败');
                }
                
                const data = await response.json();
                
                // 显示助手回复
                addMessage(data.response, 'assistant');
                
                // 播放音频
                if (data.audio_file) {
                    playAudio(data.audio_file);
                }
                
            } catch (error) {
                addMessage('抱歉，出现错误: ' + error.message, 'assistant');
            } finally {
                sendBtn.disabled = false;
                sendBtn.textContent = '发送';
            }
        }
        
        function addMessage(text, type) {
            const div = document.createElement('div');
            div.className = `message ${type}-message`;
            div.textContent = text;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        function playAudio(audioFile) {
            const div = document.createElement('div');
            div.className = 'audio-player';
            
            const audio = document.createElement('audio');
            audio.controls = true;
            audio.src = '/audio/' + audioFile;
            
            div.appendChild(audio);
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
            
            // 自动播放
            audio.play().catch(e => console.log('自动播放失败:', e));
        }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    
    # 初始化模型
    initialize_models()
    
    # 启动服务
    print(f"\n🚀 启动服务: http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
