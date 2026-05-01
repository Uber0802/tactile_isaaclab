"""Tactile encoder + ReWiND-style progress transformer.

Replaces the frozen DinoV2 image encoder of the original ReWiND with a small
trainable CNN that consumes raw tactile force fields. The downstream causal
transformer + sigmoid progress head are unchanged in spirit.
"""
from __future__ import annotations

import torch
import torch.nn as nn


_CHANNEL_PROGRESSION = {
    5: [32, 64, 128, 256],     # final per_hand_dim appended at use time
    4: [32, 64, 128],
    3: [64, 128],
    2: [128],
    1: [],
}


def _make_conv_stack(in_channels: int, num_strided_layers: int,
                     per_hand_dim: int) -> nn.Sequential:
    """Stack of `num_strided_layers` stride-2 conv blocks reaching per_hand_dim."""
    if num_strided_layers not in _CHANNEL_PROGRESSION:
        raise ValueError(f"num_strided_layers must be in {sorted(_CHANNEL_PROGRESSION)}, "
                         f"got {num_strided_layers}")
    channels = _CHANNEL_PROGRESSION[num_strided_layers] + [per_hand_dim]
    layers, in_ch = [], in_channels
    for i, out_ch in enumerate(channels):
        k, p = (7, 3) if i == 0 else (3, 1)
        layers += [
            nn.Conv2d(in_ch, out_ch, k, stride=2, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        ]
        in_ch = out_ch
    return nn.Sequential(*layers)


class TactileCNNEncoder(nn.Module):
    """Per-hand shared-weight CNN.

    Two configurations:
      * `bimanual_axis="width"`  + 5 stride-2 layers (default):
          AnyTouch2 raw layout — input (B, 2, 320, 480), split width into two
          (320, 240) hands, total stride 32.

      * `bimanual_axis="height"` + 3 stride-2 layers:
          IsaacLab-aligned layout — input (B, 2, 40, 25), split height into two
          (20, 25) hands, total stride 8.

    Input  : (B, in_channels, H, W)
    Output : (B, output_dim)
    """

    def __init__(self, in_channels: int = 2, per_hand_dim: int = 384,
                 output_dim: int = 768, num_strided_layers: int = 5,
                 bimanual_axis: str = "width"):
        super().__init__()
        if bimanual_axis not in ("width", "height"):
            raise ValueError(f"bimanual_axis must be 'width' or 'height', got {bimanual_axis!r}")
        self.per_hand_dim = per_hand_dim
        self.output_dim = output_dim
        self.bimanual_axis = bimanual_axis
        self.conv = _make_conv_stack(in_channels, num_strided_layers, per_hand_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Linear(2 * per_hand_dim, output_dim)

    def encode_one_hand(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bimanual_axis == "width":
            W = x.shape[-1]
            if W % 2 != 0:
                raise ValueError(f"expected even width to split hands, got {W}")
            left, right = x[..., : W // 2], x[..., W // 2 :]
        else:  # "height"
            H = x.shape[-2]
            if H % 2 != 0:
                raise ValueError(f"expected even height to split hands, got {H}")
            left, right = x[..., : H // 2, :], x[..., H // 2 :, :]
        l = self.encode_one_hand(left)
        r = self.encode_one_hand(right)
        return self.fusion(torch.cat([l, r], dim=-1))


class TactileReWiNDTransformer(nn.Module):
    """End-to-end tactile progress predictor.

    Forward signature matches the original ReWiNDTransformer except that
    `frames` are raw tactile tensors (B, T, 2, 320, 480) rather than
    pre-computed video embeddings.
    """

    def __init__(
        self,
        max_length: int = 16,
        text_dim: int = 384,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        per_hand_dim: int = 384,
        dropout: float = 0.1,
        num_strided_layers: int = 5,
        bimanual_axis: str = "width",
    ):
        super().__init__()
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.video_dim = 2 * per_hand_dim
        self.num_strided_layers = num_strided_layers
        self.bimanual_axis = bimanual_axis

        self.encoder = TactileCNNEncoder(
            in_channels=2,
            per_hand_dim=per_hand_dim,
            output_dim=self.video_dim,
            num_strided_layers=num_strided_layers,
            bimanual_axis=bimanual_axis,
        )

        self.video_proj = nn.Linear(self.video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Match the original: a learned bias added to the first frame only.
        self.first_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Causal mask covering [text_token, frame_0, ..., frame_{T-1}].
        causal = nn.Transformer.generate_square_subsequent_mask(self.max_length + 1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        B, T = frames.shape[:2]
        x = frames.reshape(B * T, *frames.shape[2:])
        x = self.encoder(x)
        return x.reshape(B, T, -1)

    def forward(self, frames: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 2, H, W); text_embed: (B, text_dim).

        Returns per-frame progress in [0, 1] of shape (B, T, 1).
        """
        T = frames.shape[1]
        if T > self.max_length:
            raise ValueError(f"sequence length {T} exceeds max_length {self.max_length}")

        video_h = self.video_proj(self.encode_frames(frames))      # (B, T, H)
        text_h = self.text_proj(text_embed).unsqueeze(1)            # (B, 1, H)

        # Add first-frame positional bias without breaking autograd.
        pos_bias = torch.zeros_like(video_h)
        pos_bias[:, 0] = self.first_pos_embed
        video_h = video_h + pos_bias

        seq = torch.cat([text_h, video_h], dim=1)                   # (B, 1+T, H)
        mask = self.causal_mask[: 1 + T, : 1 + T]
        out = self.transformer(seq, mask=mask)

        return self.progress_head(out[:, 1:])                       # (B, T, 1)
