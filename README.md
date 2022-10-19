# huggingface-diffusers
Small app to-be, featuring a web UI for running diffusers

## Status
Right now no more than simplistic Python scripts stolen from
https://github.com/huggingface/diffusers.

## Notes
In order to run these a huggingface token plus granted request to use model is
required, unless models are downloaded via
```
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
```
CPU/ GPU choices here reflect relatively small GPUs, e.g. GTX 1070.

## TODO
- [ ] Use a better tokenizer in ``text-to-image-gen.py``.
- [ ] Build a simple, cool web interface using
[Gradio](https://pypi.org/project/gradio/)
or
[Streamlit](https://streamlit.io/) from PyPi. 

## Appendix - More Links
https://towardsdatascience.com/hugging-face-just-released-the-diffusers-library-846f32845e65
