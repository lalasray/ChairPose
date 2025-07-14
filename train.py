# train.py
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from motion_quantizer import MotionQuantizer
from pressure2pose import Pressure2Pose
from dataset import ChairPoseDataset  # Make sure this file exists

def load_paths_from_file(path_file):
    data_paths = []
    with open(path_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                data_paths.append(tuple(parts))
    return data_paths

def main(args):
    # Load data paths
    data_paths = load_paths_from_file(args.data_list)

    # Dataset and loader
    dataset = ChairPoseDataset(data_paths)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Models
    mq = MotionQuantizer(input_dim=198, hidden_dim=256, codebook_size=1024, code_dim=512)
    p2p = Pressure2Pose()

    optimizer = torch.optim.Adam(list(mq.parameters()) + list(p2p.parameters()), lr=1e-4)

    # Training loop
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            motion_input = batch["motion"]
            pressure_input = batch["pressure"]
            chair_points = batch["chair"]

            B, T, F = motion_input.shape
            motion_input_flat = motion_input.view(B, F, T).permute(0, 2, 1)

            recon, recon_loss, q_loss, codes = mq(motion_input_flat)

            logits = p2p(pressure_input[:, 0], chair_points, prev_token=None)
            targets = codes[:, 0]
            classification_loss = F.cross_entropy(logits, targets)

            loss = recon_loss + q_loss + classification_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_list", type=str, required=True,
                        help="Path to txt file with lines: pressure.npy,chair.npy,motion.npy")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
