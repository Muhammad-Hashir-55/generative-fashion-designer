from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ml.generator import FashionGenerator
from app.schemas.generation import GenerationRequest, GenerationResponse

app = FastAPI(
    title="Generative Fashion Designer API",
    version="0.1.0",
    description="FastAPI backend for diffusion-based textile pattern generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The diffusion pipeline is loaded lazily on first generation request so local
# development stays fast while the API contract is being built.
fashion_generator = FashionGenerator()


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "generative-fashion-designer-api",
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_pattern(payload: GenerationRequest) -> GenerationResponse:
    return GenerationResponse(
        status="accepted",
        message="Generation endpoint is wired up. Image synthesis will be implemented next.",
        request=payload,
        model_id=fashion_generator.model_id,
        device=fashion_generator.device.type,
        pipeline_loaded=fashion_generator.is_loaded,
    )
