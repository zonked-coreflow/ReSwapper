import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class R3GANBlock(nn.Module):
    """R3GAN-style residual block: 1-3-1 inverted bottleneck with grouped conv.

    No normalization layers. Stability comes from fix-up initialization
    and the R3GAN loss (RpGAN + R1 + R2 gradient penalties).

    Reference: arXiv:2501.05441 "The GAN is dead; long live the GAN!"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: float = 1.5,
        group_size: int = 16,
        num_blocks_in_stage: int = 1,
    ):
        super().__init__()
        mid_channels = int(out_channels * expansion)
        # Ensure mid_channels is divisible by group_size
        mid_channels = max(mid_channels // group_size, 1) * group_size
        num_groups = mid_channels // group_size

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1, groups=num_groups, bias=False
        )
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.downsample = None
        if stride == 2:
            self.downsample = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=False)

        self.upsample = None
        if stride == -2:  # for decoder upsampling
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Fix-up initialization
        self._fixup_init(num_blocks_in_stage)

    def _fixup_init(self, num_blocks: int):
        scale = num_blocks ** (-0.25)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv1.weight.data *= scale
        nn.init.kaiming_normal_(self.conv2.weight)
        self.conv2.weight.data *= scale
        # Zero-init last conv in residual path
        nn.init.zeros_(self.conv3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        elif self.upsample is not None:
            x = self.upsample(x)

        residual = self.shortcut(x)
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.conv3(out)
        return residual + out


class CrossAttentionBlock(nn.Module):
    """Cross-attention for identity injection. Queries from spatial features,
    keys/values from identity tokens.

    Uses Flash Attention when available (PyTorch 2.0+).
    Reference: IP-Adapter / DreamID style decoupled cross-attention.
    """

    def __init__(self, dim: int, id_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.norm = nn.LayerNorm(dim)
        self.norm_id = nn.LayerNorm(id_dim)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(id_dim, dim, bias=False)
        self.to_v = nn.Linear(id_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        # Zero-init output projection for residual-friendly init
        nn.init.zeros_(self.to_out.weight)

    def forward(self, x: torch.Tensor, id_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: spatial features [B, C, H, W]
            id_tokens: identity tokens [B, N, id_dim]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        residual = x

        # Reshape spatial to sequence: [B, H*W, C]
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_seq = self.norm(x_seq)
        id_tokens = self.norm_id(id_tokens)

        q = self.to_q(x_seq)  # [B, H*W, dim]
        k = self.to_k(id_tokens)  # [B, N, dim]
        v = self.to_v(id_tokens)  # [B, N, dim]

        # Reshape for multi-head attention
        q = q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (uses Flash Attention when available)
        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.to_out(out)

        # Reshape back to spatial
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return residual + out


class AdaINIDInjection(nn.Module):
    """Adaptive Instance Normalization for identity injection at high resolutions.

    Lightweight alternative to cross-attention — projects identity embedding
    to per-channel scale and shift parameters.
    """

    def __init__(self, channels: int, id_dim: int = 512):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.fc = nn.Linear(id_dim, channels * 2)
        # Initialize to identity transform
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, id_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: feature map [B, C, H, W]
            id_embedding: identity embedding [B, id_dim]
        Returns:
            [B, C, H, W]
        """
        params = self.fc(id_embedding)  # [B, 2*C]
        scale, shift = params.chunk(2, dim=1)  # each [B, C]
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        out = self.norm(x)
        return out * (1 + scale) + shift
