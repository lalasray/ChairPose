# demo.py
import torch
import numpy as np
from motion_quantizer import MotionQuantizer
from pressure2pose import Pressure2Pose
from inference import generate_sequence
from metrics import mpjpe, smoothness_loss

# Load dummy input
pressure_seq = torch.randn(1, 10, 80, 28)
chair_points = torch.randn(1, 5000, 3)
gt_sequence = torch.randn(1, 10, 132)  # Optional: ground truth motion

# Load models
mq = MotionQuantizer(input_dim=132, hidden_dim=256, codebook_size=1024, code_dim=512)
p2p = Pressure2Pose()
mq.eval()
p2p.eval()

# Generate sequence
pred_motion, pred_tokens = generate_sequence(pressure_seq, chair_points, mq, p2p, seq_len=10)

# Optionally reshape to joints [B, T, J, 3]
pred_joints = pred_motion.view(1, 10, 44, 3)
gt_joints = gt_sequence.view(1, 10, 44, 3)

# Metrics
print("MPJPE:", mpjpe(pred_joints, gt_joints))
print("Smoothness:", smoothness_loss(pred_motion))
