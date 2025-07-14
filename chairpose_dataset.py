import torch
import numpy as np
from torch.utils.data import Dataset

class ChairPoseDataset(Dataset):
    def __init__(self, data_paths, sequence_len=15):
        self.data = data_paths  # list of tuples: (pressure_file, chair_file, motion_file)
        self.sequence_len = sequence_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pressure_file, chair_file, motion_file = self.data[idx]

        pressure = np.load(pressure_file)  # shape: (N, 80, 28)
        motion = np.load(motion_file)      # shape: (N, 22, 3)
        chair = np.load(chair_file)        # shape: (5000, 3)

        start = 0
        end = start + self.sequence_len
        motion_chunk = motion[start:end]          # (15, 22, 3)
        pressure_chunk = pressure[start:end]      # (15, 80, 28)

        # Compute motion descriptors
        pos = motion_chunk
        vel = np.gradient(pos, axis=0)
        acc = np.gradient(vel, axis=0)

        motion_feat = np.concatenate([pos, vel, acc], axis=-1)  # (15, 22, 9)
        motion_feat = motion_feat.reshape(-1, 22 * 9)            # (15, 198)

        return {
            "pressure": torch.tensor(pressure_chunk, dtype=torch.float),     # (15, 80, 28)
            "motion": torch.tensor(motion_feat, dtype=torch.float),          # (15, 198)
            "chair": torch.tensor(chair, dtype=torch.float),                # (5000, 3)
        }
