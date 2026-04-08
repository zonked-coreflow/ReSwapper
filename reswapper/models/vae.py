import torch
import torch.nn as nn


class FluxVAE(nn.Module):
    """Wrapper for the FLUX VAE (same as Z-Image-Turbo).

    16 latent channels, 8x spatial compression.
    Frozen — only used for encode/decode, no gradients.

    Load from either:
        - Tongyi-MAI/Z-Image-Turbo (HuggingFace)
        - black-forest-labs/FLUX.1-dev (HuggingFace)
        - Any diffusers AutoencoderKL with 16 latent channels
    """

    def __init__(self, pretrained_path: str = "black-forest-labs/FLUX.1-dev"):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.vae = None
        self.scaling_factor = None

    def load(self, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """Load the VAE from HuggingFace. Call this explicitly to control when the large model loads."""
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_path,
            subfolder="vae",
            torch_dtype=dtype,
        ).to(device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.scaling_factor = self.vae.config.scaling_factor

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space images to latent space.

        Args:
            images: [B, 3, H, W] in [0, 1] range
        Returns:
            latent: [B, 16, H//8, W//8]
        """
        # VAE expects [-1, 1] range
        x = images * 2.0 - 1.0
        latent_dist = self.vae.encode(x).latent_dist
        latent = latent_dist.sample()
        return latent * self.scaling_factor

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to pixel-space images.

        Args:
            latent: [B, 16, H, W]
        Returns:
            images: [B, 3, H*8, W*8] in [0, 1] range
        """
        latent = latent / self.scaling_factor
        decoded = self.vae.decode(latent).sample
        # Convert from [-1, 1] to [0, 1]
        return (decoded + 1.0) / 2.0
