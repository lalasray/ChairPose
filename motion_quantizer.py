# motion_quantizer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class VQEmbeddingEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(self.embedding.weight.clone())
        self.decay = decay
        self.eps = eps

    def forward(self, inputs):
        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (flat_input ** 2).sum(1, keepdim=True) - 2 * flat_input @ self.embedding.weight.T + \
                    (self.embedding.weight ** 2).sum(1)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(inputs.dtype)

        # Quantize
        quantized = F.embedding(encoding_indices, self.embedding.weight).view(input_shape)

        if self.training:
            # EMA update
            self.ema_cluster_size = self.decay * self.ema_cluster_size + \
                                    (1 - self.decay) * encodings.sum(0)

            dw = encodings.T @ flat_input
            self.ema_w.data = self.decay * self.ema_w + (1 - self.decay) * dw

            n = self.ema_cluster_size.sum()
            cluster_size = ((self.ema_cluster_size + self.eps) /
                            (n + self.num_embeddings * self.eps)) * n
            self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)

        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # Straight-through
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices.view(input_shape[:-1])


class MotionQuantizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, codebook_size, code_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, code_dim, kernel_size=3, padding=1)
        )
        self.vq = VQEmbeddingEMA(num_embeddings=codebook_size, embedding_dim=code_dim)
        self.decoder = nn.Sequential(
            nn.Conv1d(code_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: [B, T*J, D] → [B, D, T*J]
        x = x.permute(0, 2, 1)
        z_e = self.encoder(x)
        z_e = z_e.permute(0, 2, 1)  # [B, T*J, code_dim]
        quantized, q_loss, codes = self.vq(z_e)
        quantized = quantized.permute(0, 2, 1)
        recon = self.decoder(quantized)
        recon = recon.permute(0, 2, 1)  # [B, T*J, D]
        recon_loss = F.mse_loss(recon, x.permute(0, 2, 1))
        return recon, recon_loss, q_loss, codes
