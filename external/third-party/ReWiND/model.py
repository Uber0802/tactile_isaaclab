import torch
import torch.nn as nn


class ReWiNDTransformer(nn.Module):
    def __init__(self, args, video_dim=768, text_dim=384, hidden_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.args = args

        # Project video and text to common dimension
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Position embeddings for video sequence
        self.first_pos_embed = nn.Parameter(torch.randn(1, hidden_dim))  # 32 is max_length

        # Class token embedding
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Progress prediction head (applied to each frame)
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        # Attention mask for causal self-attention. Built once on cuda; the
        # forward pass moves it to the correct device per-call if needed.
        self.attention_mask = nn.Transformer.generate_square_subsequent_mask(
            args.max_length + 1
        ).to("cuda")

    def forward(self, video_frames, text_embed):
        batch_size = video_frames.shape[0]

        # Project inputs to common dimension
        video_embed = self.video_proj(video_frames)             # [B, seq_len, hidden]
        text_embed = self.text_proj(text_embed).unsqueeze(1)    # [B, 1, hidden]

        # Add positional embedding to the FIRST video frame only.
        video_embed[:, 0] += self.first_pos_embed

        # Combine sequence: [text, video_frames]
        sequence = torch.cat([text_embed, video_embed], dim=1)

        seq_len = sequence.size(1)
        mask = self.attention_mask[:seq_len, :seq_len].to(sequence.device)
        transformed = self.transformer(sequence, is_causal=True, mask=mask)

        # Per-frame progress predictions (drop the text token at position 0).
        progress_preds = self.progress_head(transformed[:, 1:])

        return progress_preds
