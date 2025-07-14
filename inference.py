# inference.py
import torch
from motion_quantizer import MotionQuantizer
from pressure2pose import Pressure2Pose

@torch.no_grad()
def generate_sequence(pressure_seq, chair_points, mq: MotionQuantizer, p2p: Pressure2Pose, seq_len=10):
    device = next(p2p.parameters()).device
    mq.eval()
    p2p.eval()

    B = pressure_seq.size(0)
    prev_token = None
    token_outputs = []
    recon_poses = []

    for t in range(seq_len):
        pressure_t = pressure_seq[:, t]  # [B, 80, 28]

        logits = p2p(pressure_t.to(device), chair_points.to(device), prev_token)
        probs = torch.softmax(logits, dim=-1)
        predicted_token = torch.argmax(probs, dim=-1)  # [B]

        # Quantized embedding lookup from MQ’s codebook
        token_embedding = mq.vq.embedding(predicted_token)  # [B, D]
        token_embedding = token_embedding.unsqueeze(-1)  # [B, D, 1]
        decoded = mq.decoder(token_embedding).squeeze(-1)  # [B, F]

        token_outputs.append(predicted_token.cpu())
        recon_poses.append(decoded.cpu())

        prev_token = token_embedding.squeeze(-1)

    recon_sequence = torch.stack(recon_poses, dim=1)  # [B, seq_len, F]
    return recon_sequence, torch.stack(token_outputs, dim=1)  # poses, token indices
