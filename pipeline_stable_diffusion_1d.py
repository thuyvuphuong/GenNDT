import torch
from torch import nn
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class StableDiffusion1DPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: nn.Module,
        unet: nn.Module,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        scheduler: DDPMScheduler,
    ):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.register_modules(
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )

    def encode_prompt(self, prompt, device, guidance_scale):
        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)

        # Get embeddings
        prompt_embeds = self.text_encoder(input_ids)[0]

        if guidance_scale > 1.0:
            # Unconditional prompt (empty)
            uncond_input = self.tokenizer(
                ["" for _ in range(input_ids.shape[0])],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            uncond_ids = uncond_input.input_ids.to(device)
            negative_prompt_embeds = self.text_encoder(uncond_ids)[0]
        else:
            negative_prompt_embeds = None

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        signal_length=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        generator=None,
        latents=None,
        return_dict=True,
        output_type="np",
    ):
        device = self._execution_device
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # 1. Determine latent/signal length
        if signal_length is None:
            if isinstance(self.unet.config, dict):
                latent_length = self.unet.config.get("sample_size", 64)
            else:
                latent_length = getattr(self.unet.config, "sample_size", 64)
            signal_length = latent_length * self.vae_scale_factor
        else:
            latent_length = signal_length // self.vae_scale_factor

        # 2. Encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, device, batch_size, guidance_scale > 1.0, negative_prompt
        )
        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare latents
        if latents is None:
            latents = torch.randn(
                (batch_size, self.unet.config["in_channels"], latent_length),
                generator=generator, device=device, dtype=prompt_embeds.dtype
            )
        latents *= self.scheduler.init_noise_sigma

        # 5. Denoising loop
        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            t_batch = torch.full((latent_model_input.shape[0],), t, device=latent_model_input.device, dtype=torch.long)
            noise_pred = self.unet(latent_model_input, t_batch, prompt_embeds)[0]
            if guidance_scale > 1.0:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 6. Decode
        signal = self.vae.decode(latents)[0]  # [B, C, L]
        if output_type == "np":
            signal = signal.squeeze(1).detach().cpu().numpy()
        if return_dict:
            return {"signals": signal}
        return signal