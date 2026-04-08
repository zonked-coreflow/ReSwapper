import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import R3GANBlock, CrossAttentionBlock, AdaINIDInjection


class IdentityTokenGenerator(nn.Module):
    """Projects a frozen identity embedding into multiple tokens for cross-attention."""

    def __init__(self, embedding_dim: int = 512, num_tokens: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.tokenizer = nn.Linear(embedding_dim, num_tokens * embedding_dim)
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: [B, embedding_dim] from frozen identity encoder
        Returns:
            tokens: [B, num_tokens, embedding_dim]
        """
        x = self.mlp(embedding)
        tokens = self.tokenizer(x)
        return tokens.view(-1, self.num_tokens, self.embedding_dim)


class FaceSwapGenerator(nn.Module):
    """Latent-space U-Net generator with R3GAN blocks and cross-attention identity injection.

    Operates on FLUX VAE latents (16 channels, 64x64 for 512px images).
    Includes occlusion-aware swap mask prediction.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        base_channels: int = 256,
        bottleneck_channels: int = 512,
        id_dim: int = 512,
        num_id_tokens: int = 16,
        parsing_channels: int = 19,
        group_size: int = 16,
        expansion: float = 1.5,
        num_bottleneck_blocks: int = 2,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        bc = base_channels
        bnc = bottleneck_channels

        # Parsing map fusion: 19-class segmentation -> feature space
        self.parse_embed = nn.Sequential(
            nn.Conv2d(parsing_channels, 64, 3, padding=1),
            nn.SiLU(),
        )

        # Identity token generator
        self.id_token_gen = IdentityTokenGenerator(id_dim, num_id_tokens)

        # Encoder
        self.enc0 = R3GANBlock(latent_channels + 64, bc, stride=1, expansion=expansion, group_size=group_size, num_blocks_in_stage=4)
        self.enc1 = R3GANBlock(bc, bc, stride=2, expansion=expansion, group_size=group_size, num_blocks_in_stage=4)
        self.enc2 = R3GANBlock(bc, bnc, stride=2, expansion=expansion, group_size=group_size, num_blocks_in_stage=4)
        self.enc3 = R3GANBlock(bnc, bnc, stride=2, expansion=expansion, group_size=group_size, num_blocks_in_stage=4)

        # Bottleneck
        self.bottleneck_blocks = nn.ModuleList()
        for _ in range(num_bottleneck_blocks):
            self.bottleneck_blocks.append(
                R3GANBlock(bnc, bnc, stride=1, expansion=expansion, group_size=group_size, num_blocks_in_stage=num_bottleneck_blocks)
            )
        self.bottleneck_attn = CrossAttentionBlock(bnc, id_dim, num_heads=8)

        # Decoder with skip connections and identity injection
        # dec3: 8x8 -> 16x16, cross-attention
        self.dec3_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3_block = R3GANBlock(bnc + bnc, bnc, stride=1, expansion=expansion, group_size=group_size, num_blocks_in_stage=4)
        self.dec3_attn = CrossAttentionBlock(bnc, id_dim, num_heads=8)

        # dec2: 16x16 -> 32x32, cross-attention
        self.dec2_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2_block = R3GANBlock(bnc + bc, bc, stride=1, expansion=expansion, group_size=group_size, num_blocks_in_stage=4)
        self.dec2_attn = CrossAttentionBlock(bc, id_dim, num_heads=8)

        # dec1: 32x32 -> 64x64, AdaIN (lightweight at high res)
        self.dec1_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1_block = R3GANBlock(bc + bc, bc, stride=1, expansion=expansion, group_size=group_size, num_blocks_in_stage=4)
        self.dec1_adain = AdaINIDInjection(bc, id_dim)

        # Output heads
        self.latent_head = nn.Conv2d(bc, latent_channels, 3, padding=1)
        self.mask_head = nn.Sequential(
            nn.Conv2d(bc, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        target_latent: torch.Tensor,
        id_embedding: torch.Tensor,
        parsing_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target_latent: [B, 16, H, W] FLUX VAE latent of target face
            id_embedding: [B, 512] frozen identity embedding from source face
            parsing_map: [B, 19, H, W] face parsing map of target face
        Returns:
            swapped_latent: [B, 16, H, W] swapped face latent
            swap_mask: [B, 1, H, W] occlusion-aware swap mask (1=swap, 0=keep target)
        """
        # Generate identity tokens for cross-attention
        id_tokens = self.id_token_gen(id_embedding)  # [B, 16, 512]

        # Fuse parsing map
        parse_feat = self.parse_embed(parsing_map)  # [B, 64, H, W]

        # Concatenate latent + parsing features
        x = torch.cat([target_latent, parse_feat], dim=1)  # [B, 16+64, H, W]

        # Encoder
        if self.gradient_checkpointing and self.training:
            e0 = checkpoint(self.enc0, x, use_reentrant=False)
            e1 = checkpoint(self.enc1, e0, use_reentrant=False)
            e2 = checkpoint(self.enc2, e1, use_reentrant=False)
            e3 = checkpoint(self.enc3, e2, use_reentrant=False)
        else:
            e0 = self.enc0(x)   # [B, bc, 64, 64]
            e1 = self.enc1(e0)  # [B, bc, 32, 32]
            e2 = self.enc2(e1)  # [B, bnc, 16, 16]
            e3 = self.enc3(e2)  # [B, bnc, 8, 8]

        # Bottleneck
        h = e3
        for block in self.bottleneck_blocks:
            h = block(h)
        h = self.bottleneck_attn(h, id_tokens)

        # Decoder
        # dec3: 8->16
        h = self.dec3_up(h)
        h = torch.cat([h, e2], dim=1)
        h = self.dec3_block(h)
        h = self.dec3_attn(h, id_tokens)

        # dec2: 16->32
        h = self.dec2_up(h)
        h = torch.cat([h, e1], dim=1)
        h = self.dec2_block(h)
        h = self.dec2_attn(h, id_tokens)

        # dec1: 32->64
        h = self.dec1_up(h)
        h = torch.cat([h, e0], dim=1)
        h = self.dec1_block(h)
        h = self.dec1_adain(h, id_embedding)

        # Output heads
        swapped_latent = self.latent_head(h)
        swap_mask = self.mask_head(h)

        return swapped_latent, swap_mask
