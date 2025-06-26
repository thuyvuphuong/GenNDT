import torch
import inspect
from torch import nn
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Any, Callable, Dict, List, Optional, Union

def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
    ):
        r"""
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
                must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
                `num_inference_steps` and `sigmas` must be `None`.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps


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
    
    @property
    def interrupt(self):
        return self._interrupt
    
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        signal_length=None,
        latent_length=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=None,
        latents=None,
        return_dict=True,
        output_type="np",
        sigmas: List[float] = None,
        timesteps: List[int] = None,
    ):
        device = self._execution_device
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        self._interrupt = False

        # 1. Determine latent/signal length
        if signal_length is None:
            if isinstance(self.unet.config, dict):
                latent_length = self.vae.config.get("latent_channels", 64)
            else:
                latent_length = getattr(self.vae.config, "latent_channels", 64)
            signal_length = getattr(self.vae.config, "input_length", 500)
        else:
            latent_length = latent_length
            signal_length = signal_length

        # 2. Encode prompt
        
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, device, guidance_scale=guidance_scale
        )
        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 3. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 4. Prepare latents
        if latents is None:
            latents = randn_tensor((batch_size, self.unet.config["in_channels"], latent_length), 
                                   generator=generator, device=device, dtype=prompt_embeds.dtype)
        else:
            latents = latents.to(device)
        latents *= self.scheduler.init_noise_sigma

        # 5. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                t_batch = torch.full((batch_size,), t, dtype=torch.long, device=t.device)
                noise_pred = self.unet(latent_model_input, t_batch, prompt_embeds)[0]
                if guidance_scale > 1.0:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

            # 6. Decode
            signal = self.vae.decode(latents)[0]  # [B, C, L]
            if output_type == "np":
                signal = signal.squeeze(1).detach().cpu().numpy()
            if return_dict:
                return {"signals": signal}
            return signal