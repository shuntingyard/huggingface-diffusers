from io import BytesIO
import time

from torch import autocast
import torch
import requests
import PIL

from diffusers import StableDiffusionInpaintPipeline


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

device = "cuda"
model_id_or_path = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id_or_path,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
)
pipe = pipe.to(device)

prompt = "a cat sitting on a bench"
with autocast("cuda"):
    images = pipe(
        prompt=prompt,
        init_image=init_image,
        mask_image=mask_image,
        strength=0.75,
    ).images

images[0].save(f"{int(time.time())}_cat_on_bench.png")
