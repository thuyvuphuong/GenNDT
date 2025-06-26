#%%
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers import DDPMScheduler

from unet1D import SimpleUNet1D
from autoencoder import Conv1DAutoencoder
from pipeline_stable_diffusion_1d import StableDiffusion1DPipeline

#%%
# =======================
# Load components
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = Conv1DAutoencoder.from_pretrained("genNDT_model/checkpoint-2000/vae", map_location=device)
unet = SimpleUNet1D.from_pretrained("genNDT_model/checkpoint-2000/unet", map_location=device)
text_encoder = CLIPTextModel.from_pretrained("pretrained_model/stable-diffusion-v1-4/text_encoder")
tokenizer = CLIPTokenizer.from_pretrained("pretrained_model/stable-diffusion-v1-4/tokenizer")
scheduler = DDPMScheduler.from_pretrained("pretrained_model/stable-diffusion-v1-4/scheduler")

pipe = StableDiffusion1DPipeline(
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)
pipe = pipe.to(device)

#%%
# =======================
# Construct structured prompt
# =======================
X = 40
Y = 28
L = 0.1
dia = 10
D = 0.86
r = 10
has_crack = 1
zone_id = 7

prompt = (
    f"X coordinate: {X}, Y coordinate: {Y}, Length: {L}, Diameter: {dia}, "
    f"Depth: {D}, Radius: {r}, Has crack: {has_crack}, Zone ID: {zone_id}"
)

#%%
# =======================
# Generate signal
# =======================
output = pipe(prompt, num_inference_steps=50, guidance_scale=7.5)
signal = output["signal"][0]  # shape: [L]

# %%
plt.plot(signal)

# %%
