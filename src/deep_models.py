import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(device=x.device, dtype=x.dtype)


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


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
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return nn.functional.relu(x + residual, inplace=True)


class AttentionSleepModel(nn.Module):
    def __init__(self, in_channels=2, num_classes=5, d_model=128, n_heads=8):
        super().__init__()
        self.epoch_feature_extractor = nn.Sequential(
            ConvBlock1d(in_channels, 32, dropout=0.1),
            nn.MaxPool1d(kernel_size=2),
            ConvBlock1d(32, 64, dropout=0.1),
            nn.MaxPool1d(kernel_size=2),
            ConvBlock1d(64, d_model, dropout=0.1),
            nn.MaxPool1d(kernel_size=2),
        )
        self.within_epoch_position = SinusoidalPositionalEncoding(d_model=d_model, max_len=512)
        within_epoch_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.within_epoch_transformer = nn.TransformerEncoder(within_epoch_layer, num_layers=2)
        self.epoch_attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.context_position = SinusoidalPositionalEncoding(d_model=d_model, max_len=31)
        context_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, n_heads // 2),
            dim_feedforward=2 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.context_transformer = nn.TransformerEncoder(context_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, 2 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2 * d_model, num_classes),
        )

    def encode_epoch(self, x):
        x = self.epoch_feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.within_epoch_position(x)
        x = self.within_epoch_transformer(x)
        attention_scores = torch.softmax(self.epoch_attention_pool(x), dim=1)
        return torch.sum(attention_scores * x, dim=1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        batch_size, context_epochs, in_channels, n_times = x.shape
        x = x.reshape(batch_size * context_epochs, in_channels, n_times)
        epoch_embeddings = self.encode_epoch(x)
        epoch_embeddings = epoch_embeddings.reshape(batch_size, context_epochs, -1)

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
    def __init__(self, in_channels=2, num_classes=5, base_channels=32):
        super().__init__()
        self.enc1 = ConvBlock1d(in_channels, base_channels, dropout=0.1)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock1d(base_channels, base_channels * 2, dropout=0.1)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock1d(base_channels * 2, base_channels * 4, dropout=0.1)
        self.pool3 = nn.MaxPool1d(2)

        self.bottleneck = ConvBlock1d(base_channels * 4, base_channels * 8, dropout=0.2)

        self.up3 = UpBlock1d(base_channels * 8, base_channels * 4, base_channels * 4, dropout=0.1)
        self.up2 = UpBlock1d(base_channels * 4, base_channels * 2, base_channels * 2, dropout=0.1)
        self.up1 = UpBlock1d(base_channels * 2, base_channels, base_channels, dropout=0.1)

        context_channels = base_channels * 2
        self.epoch_head = nn.Sequential(
            nn.Conv1d(base_channels, context_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(context_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.context_mixer = nn.Sequential(
            TemporalContextBlock(context_channels, dilation=1, dropout=0.1),
            TemporalContextBlock(context_channels, dilation=2, dropout=0.1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(3 * context_channels),
            nn.Linear(3 * context_channels, 2 * context_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2 * context_channels, num_classes),
        )

    def encode_epoch(self, x):
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
        return self.epoch_head(x).squeeze(-1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        batch_size, context_epochs, in_channels, n_times = x.shape
        x = x.reshape(batch_size * context_epochs, in_channels, n_times)
        epoch_embeddings = self.encode_epoch(x)
        epoch_embeddings = epoch_embeddings.reshape(batch_size, context_epochs, -1)

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
