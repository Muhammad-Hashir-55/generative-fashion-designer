from pydantic import BaseModel, ConfigDict, Field


class GenerationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(..., min_length=1, description="Text prompt for pattern generation.")
    negative_prompt: str = Field(
        default="",
        description="Prompt terms to suppress during generation.",
    )
    alpha: float = Field(
        default=1.0,
        ge=0.0,
        description="Future content-loss weight for style transfer blending.",
    )
    beta: float = Field(
        default=1.0,
        ge=0.0,
        description="Future style-loss weight for style transfer blending.",
    )


class GenerationResponse(BaseModel):
    status: str
    message: str
    request: GenerationRequest
    model_id: str
    device: str
    pipeline_loaded: bool
