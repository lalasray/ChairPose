# pressure2pose.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pressure2Pose(nn.Module):
    def __init__(self, pressure_dim=(80, 28), chair_feat_dim=256, token_dim=512, codebook_size=1024):
        super().__init__()
        pressure_flattened = pressure_dim[0] * pressure_dim[1]

        self.pressure_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pressure_flattened, 512),
            nn.ReLU(),
            nn.Linear(512, token_dim)
        )

        self.chair_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, chair_feat_dim),
            nn.ReLU()
        )

        self.point_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(token_dim + chair_feat_dim + token_dim, 512),
            nn.ReLU(),
            nn.Linear(512, codebook_size)  # logits for classification
        )

        self.start_token = nn.Parameter(torch.randn(1, token_dim))

    def forward(self, pressure, chair_points, prev_token):
        B = pressure.size(0)
        pressure_feat = self.pressure_encoder(pressure)

        chair_feat = self.chair_encoder(chair_points)  # [B, N_pts, chair_feat_dim]
        chair_feat = chair_feat.permute(0, 2, 1)  # [B, chair_feat_dim, N_pts]
        chair_feat = self.point_pool(chair_feat).squeeze(-1)  # [B, chair_feat_dim]

        if prev_token is None:
            prev_token = self.start_token.expand(B, -1)

        combined = torch.cat([pressure_feat, chair_feat, prev_token], dim=-1)
        logits = self.fc(combined)  # [B, codebook_size]
        return logits
