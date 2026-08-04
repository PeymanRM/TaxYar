import json
import time
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.runnables import Runnable
from rag.pipeline import get_rag_chain

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
    model: str = "taxyar"
    messages: list[Message]
    stream: bool = True

@app.post("/chat/completions")
async def chat(
    req: ChatRequest,
    chain: Runnable = Depends(get_rag_chain)
    ):
    # print(f"Received req: {req}")
    # TODO Manage History
    query = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        ""
    )

    async def generate():
        async for chunk in chain.astream(query):
            if "answer" in chunk and chunk["answer"]:
                content = chunk["answer"]
                # print(f"Yielding chunk: {content}")
                
                # Format chunk as an OpenAI SSE object
                payload = {
                    "id": "chatcmpl-taxyar",
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
            "id": "chatcmpl-taxyar",
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