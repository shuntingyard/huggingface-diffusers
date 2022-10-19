import time

from torch import autocast
from diffusers import DiffusionPipeline

device = "cuda"  # Not working for memory reasons.
# (Cpu not working RuntimeError: expected scalar type BFloat16 but found Float)
model_id = "CompVis/ldm-text2im-large-256"

# load model and scheduler
ldm = DiffusionPipeline.from_pretrained(model_id)
ldm = ldm.to(device)

# run pipeline in inference (sample random noise and denoise)
prompt = "A painting of a squirrel eating a burger"
with autocast(device):
    image = ldm(
        [prompt], num_inference_steps=50, eta=0.3, guidance_scale=6
    ).images[0]

# save image
image.save(f"{int(time.time())}_squirrel.png")
