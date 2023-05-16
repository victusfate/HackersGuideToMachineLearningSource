# dependencies
# pip install transformers
# pip install diffuser
# pip install torch
# or https://colab.research.google.com/gist/victusfate/a6f6b7af0548240da88e92d7cd38ca75/-stable-diffusion-v1-5.ipynb

import transformers
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
# display(image)    
image.save("astronaut_rides_horse.png")
