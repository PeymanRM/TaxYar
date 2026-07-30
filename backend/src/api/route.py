import json
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.rag.pipeline import rag_chain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "your-model-id"
    messages: list[Message]
    stream: bool = True

@app.post("/chat/completions")
async def chat(req: ChatRequest):
    print(f"Received req: {req}")
    query = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        ""
    )

    async def generate():
        async for chunk in rag_chain.astream(query):
            if "answer" in chunk and chunk["answer"]:
                content = chunk["answer"]
                # print(f"Yielding chunk: {content}")
                
                # Format chunk as an OpenAI SSE object
                payload = {
                    "id": "chatcmpl-rag",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"

        # Send final stop chunk
        stop_payload = {
            #TODO
            "id": "chatcmpl-rag",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(stop_payload)}\n\n"
        # OpenAI stream termination signal
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")