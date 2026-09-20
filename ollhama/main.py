
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Ollama Local API",
    description="API para executar modelos locais via Ollama",
    version="1.0.0",
)

OLLAMA_URL = "http://localhost:11434"

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = "qwen3:14b"
    system: Optional[str] = None
    format: Optional[str | dict] = None


class GenerateResponse(BaseModel):
    model: str
    response: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@app.get("/")
def root():
    return {"message": "Ollama API funcionando"}


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.get(
                f"{OLLAMA_URL}/api/tags"
            )
            response.raise_for_status()

        return {"status": "ok", "ollama": "connected"}

    except httpx.HTTPError:
        raise HTTPException(
            status_code=503,
            detail="Não foi possível conectar ao Ollama",
        )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    messages = []

    if request.system:
        messages.append({
            "role": "system",
            "content": request.system,
        })

    messages.append({
        "role": "user",
        "content": request.prompt,
    })

    payload = {
        "model": request.model,
        "messages": messages,
        "stream": False,
        "think": False,
    }
    if request.format is not None:
        payload["format"] = request.format

    try:
        payload["options"] = {
            **payload.get("options", {}),
            "num_ctx": 40960,
            "temperature": 0.0,
        }
        
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()

        return GenerateResponse(
            model=data["model"],
            response=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_tokens=(
                data.get("prompt_eval_count", 0)
                + data.get("eval_count", 0)
                if data.get("prompt_eval_count") is not None
                and data.get("eval_count") is not None
                else None
            ),
        )

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Erro retornado pelo Ollama: {exc.response.text}",
        )

    except httpx.HTTPError:
        raise HTTPException(
            status_code=503,
            detail="Erro ao conectar com o Ollama",
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)