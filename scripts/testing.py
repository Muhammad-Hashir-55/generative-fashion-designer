import os
import replicate
from dotenv import load_dotenv

# Get API token from environment variable
load_dotenv()  # Loads variables from .env file
replicate.api_token = os.getenv("REPLICATE_API_TOKEN")

# Optional: Add error handling if token is missing
if not replicate.api_token:
    raise ValueError("REPLICATE_API_TOKEN environment variable not set")

output = replicate.run(
    "black-forest-labs/flux-2-pro",
    input={
        "prompt": "Glossy candy-colored 3D letters in hot pink, electric orange, and lime green on a sun-drenched poolside patio with bold terrazzo tiles and vintage lounge chairs in turquoise and yellow. Shot on Kodachrome film with a Hasselblad 500C, warm golden afternoon sunlight, dramatic lens flare, punchy oversaturated colors with that distinctive 70s yellow-orange cast, shallow depth of field with the text sharp in the foreground, tropical palms and a sparkling aquamarine pool behind that spells out \"Run FLUX.2 [pro] on Replicate!\"",
        "resolution": "1 MP",
        "aspect_ratio": "1:1",
        "input_images": [],
        "output_format": "webp",
        "output_quality": 80,
        "safety_tolerance": 2
    }
)

# Access the file URL
print(output.url)

# Write the file to disk
with open("my-image.webp", "wb") as file:
    file.write(output.read())