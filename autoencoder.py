import torch
import torch.nn as nn
import os
import json

class Conv1DAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, latent_channels=64, input_length=500):
        super().__init__()
        # Save config for reproducibility
        self.config = {
            "in_channels": in_channels,
            "base_channels": base_channels,
            "latent_channels": latent_channels,
            "input_length": input_length,
        }
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),   # [B,16,250]
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),  # [B,32,125]
            nn.ReLU(),
            nn.Conv1d(base_channels*2, latent_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(latent_channels, latent_channels, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, base_channels*2, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # Save weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f)

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