import torch
import torch.nn as nn
import math
import os
import json

# Timestep embedding like in SD
class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        half_dim = self.linear1.in_features // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return self.linear2(self.act(self.linear1(emb)))


class CrossAttentionBlock(nn.Module):
    def __init__(self, latent_dim, context_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            kdim=context_dim,
            vdim=context_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, context):
        # x: (B, C, L) → (B, L, C)
        x = x.permute(0, 2, 1)
        x_norm = self.norm(x)
        attn_output, _ = self.attn(x_norm, context, context)
        x = x + self.proj(attn_output)
        return x.permute(0, 2, 1)


class SimpleUNet1D(nn.Module):
    def __init__(self, in_channels=64, context_dim=768, base_channels=128):
        super().__init__()
        self.config = {
            "in_channels": in_channels,
            "context_dim": context_dim,
            "base_channels": base_channels,
        }
        self.time_embed = TimestepEmbedding(in_channels)

        self.down = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )

        self.mid_attn = CrossAttentionBlock(base_channels, context_dim)

        self.up = nn.Sequential(
            nn.ConvTranspose1d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, in_channels, 3, padding=1),
        )

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
            
    @classmethod
    def from_pretrained(cls, load_directory, map_location=None):
        # Load config
        config_path = os.path.join(load_directory, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)
        # Load weights
        weight_path = os.path.join(load_directory, "pytorch_model.bin")
        model.load_state_dict(torch.load(weight_path, map_location=map_location or "cpu"))
        return model

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, x, timesteps, encoder_hidden_states, return_dict=False):
        t_emb = self.time_embed(timesteps)  # [B, C]
        t_emb = t_emb[:, :, None].expand(-1, -1, x.shape[2])  # [B, C, L]
        x = x + t_emb

        h = self.down(x)                             # [B, base_channels, L/2]
        h = self.mid_attn(h, encoder_hidden_states)  # cross-attn
        h = self.up(h)                               # [B, in_channels, L]

        if return_dict:
            return {"sample": h}
        return (h,)