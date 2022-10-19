import time

# make sure you're logged in with `huggingface-cli login`
from torch import autocast, float16
from diffusers import StableDiffusionPipeline

use_cuda = True

if use_cuda:
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=float16,
        use_auth_token=True,
    )
    pipe = pipe.to("cuda")
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", use_auth_token=True
    )

prompt = "Elon Musk holding his own beer"
with autocast("cuda"):
    image = pipe(prompt).images[0]

IMG_TYPE = ".png"
image.save(f"{int(time.time())}_{prompt.replace(' ', '_')}{IMG_TYPE}")
