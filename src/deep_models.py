import math

import torch
from torch import nn


def make_group_norm(num_channels, max_groups=8):
    num_groups = min(max_groups, num_channels)
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build_encoding(max_len), persistent=False)

    def _build_encoding(self, max_len):
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(max_len, self.d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            self.pe = self._build_encoding(x.size(1)).to(device=x.device, dtype=x.dtype)
        return x + self.pe[:, : x.size(1)].to(device=x.device, dtype=x.dtype)


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.norm1 = make_group_norm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.norm2 = make_group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.functional.gelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return nn.functional.gelu(x + residual)


class TemporalContextBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm1 = make_group_norm(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm2 = make_group_norm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.functional.gelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return nn.functional.gelu(x + residual)


class AttentionSleepModel(nn.Module):
    def __init__(self, in_channels=2, num_classes=5, d_model=96, n_heads=4):
        super().__init__()
        self.epoch_feature_extractor = nn.Sequential(
            ConvBlock1d(in_channels, 24, dropout=0.1),
            nn.MaxPool1d(kernel_size=2),
            ConvBlock1d(24, 48, dropout=0.1),
            nn.MaxPool1d(kernel_size=2),
            ConvBlock1d(48, d_model, dropout=0.1),
            nn.MaxPool1d(kernel_size=2),
        )
        self.within_epoch_position = SinusoidalPositionalEncoding(d_model=d_model, max_len=4096)
        epoch_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.within_epoch_transformer = nn.TransformerEncoder(epoch_encoder_layer, num_layers=1)
        self.epoch_summary = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.context_position = SinusoidalPositionalEncoding(d_model=d_model, max_len=31)
        context_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, n_heads // 2),
            dim_feedforward=2 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.context_transformer = nn.TransformerEncoder(context_encoder_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2 * d_model, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        batch_size, context_epochs, in_channels, n_times = x.shape
        x = x.reshape(batch_size * context_epochs, in_channels, n_times)
        x = self.epoch_feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.within_epoch_position(x)
        x = self.within_epoch_transformer(x)
        epoch_embeddings = self.epoch_summary(x.mean(dim=1))
        epoch_embeddings = epoch_embeddings.view(batch_size, context_epochs, -1)

        context_embeddings = self.context_position(epoch_embeddings)
        context_embeddings = self.context_transformer(context_embeddings)

        center_embedding = context_embeddings[:, context_epochs // 2]
        global_embedding = context_embeddings.mean(dim=1)
        combined = torch.cat(
            [
                center_embedding,
                global_embedding,
                center_embedding - global_embedding,
            ],
            dim=-1,
        )
        return self.classifier(combined)


class UpBlock1d(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = ConvBlock1d(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            if diff > 0:
                x = nn.functional.pad(x, (0, diff))
            elif diff < 0:
                x = x[..., : skip.size(-1)]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class USleepModel(nn.Module):
    def __init__(self, in_channels=2, num_classes=5, base_channels=24):
        super().__init__()
        self.enc1 = ConvBlock1d(in_channels, base_channels, dropout=0.1)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock1d(base_channels, base_channels * 2, dropout=0.1)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock1d(base_channels * 2, base_channels * 4, dropout=0.1)
        self.pool3 = nn.MaxPool1d(2)

        self.bottleneck = ConvBlock1d(base_channels * 4, base_channels * 6, dropout=0.15)

        self.up3 = UpBlock1d(base_channels * 6, base_channels * 4, base_channels * 4, dropout=0.1)
        self.up2 = UpBlock1d(base_channels * 4, base_channels * 2, base_channels * 2, dropout=0.1)
        self.up1 = UpBlock1d(base_channels * 2, base_channels, base_channels, dropout=0.1)

        context_channels = base_channels * 2
        self.epoch_head = nn.Sequential(
            nn.Conv1d(base_channels, context_channels, kernel_size=3, padding=1),
            make_group_norm(context_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.context_mixer = nn.Sequential(
            TemporalContextBlock(context_channels, dilation=1, dropout=0.1),
            TemporalContextBlock(context_channels, dilation=2, dropout=0.1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(3 * context_channels),
            nn.Linear(3 * context_channels, 2 * context_channels),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2 * context_channels, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        batch_size, context_epochs, in_channels, n_times = x.shape
        x = x.reshape(batch_size * context_epochs, in_channels, n_times)

        skip1 = self.enc1(x)
        x = self.pool1(skip1)
        skip2 = self.enc2(x)
        x = self.pool2(skip2)
        skip3 = self.enc3(x)
        x = self.pool3(skip3)

        x = self.bottleneck(x)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        epoch_embeddings = self.epoch_head(x).squeeze(-1)
        epoch_embeddings = epoch_embeddings.view(batch_size, context_epochs, -1)

        context_embeddings = epoch_embeddings.transpose(1, 2)
        context_embeddings = self.context_mixer(context_embeddings)
        context_embeddings = context_embeddings.transpose(1, 2)

        center_embedding = context_embeddings[:, context_epochs // 2]
        global_embedding = context_embeddings.mean(dim=1)
        combined = torch.cat(
            [
                center_embedding,
                global_embedding,
                center_embedding - global_embedding,
            ],
            dim=-1,
        )
        return self.classifier(combined)
