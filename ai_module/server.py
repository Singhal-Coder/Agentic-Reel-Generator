from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from .app.graph import app as graph_app
from .app.schemas.user_params import UserParams


class GenerateRequest(BaseModel):
    prompt: str
    params: UserParams


app = FastAPI(title="Agentic Reel Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    # Map incoming UserParams (vduration_sec) to pipeline params (duration_sec)
    pipeline_params = {}
    if req.params.vduration_sec is not None:
        pipeline_params["duration_sec"] = req.params.vduration_sec
    if req.params.aspect_ratio is not None:
        pipeline_params["aspect_ratio"] = req.params.aspect_ratio
    if req.params.voice_style is not None:
        pipeline_params["voice_style"] = req.params.voice_style
    if req.params.music_style is not None:
        pipeline_params["music_style"] = req.params.music_style

    initial_input = {
        "prompt": req.prompt,
        "params": pipeline_params,
    }

    final_state = graph_app.invoke(initial_input)
    output_path = final_state.get("output_path")

    def iterfile(path, chunk_size: int = 1024 * 1024):
        with open(path, "rb") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        iterfile(str(output_path)),
        media_type="video/mp4",
        headers={"Content-Disposition": "inline; filename=generated_reel.mp4"}
    )


