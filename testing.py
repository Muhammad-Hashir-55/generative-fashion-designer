import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="auto",
    api_key=os.getenv("HF_TOKEN"),
)

# Use the base model name
image = client.text_to_image(
    "Astronaut riding a horse",
    model="black-forest-labs/FLUX.1-dev",
)

image.save("output.png")