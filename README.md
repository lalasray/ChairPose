# ChairPose

**ChairPose** is a framework for estimating full-body 3D seated poses from pressure sensor data on arbitrary chairs, using a discrete motion representation and pressure-conditioned generative modeling.

---

## Features

- **MotionQuantizer**: Compresses 3D motion into discrete tokens via Vector Quantized VAE.
- **Pressure2Pose**: Autoregressive classifier predicting pose tokens from pressure and chair geometry.
- Sequence generation with autoregressive decoding.
- Evaluation metrics: MPJPE (Mean Per Joint Position Error) and smoothness.
- End-to-end inference/demo support.

---

## File Structure

- motion_quantizer.py # VQ-VAE for motion quantization
- pressure2pose.py # Autoregressive pose prediction model
- train.py # Training loop
- inference.py # Autoregressive sequence generation
---

## Setup

1. Clone repository and install dependencies:

```bash
git clone https://github.com/your-repo/chairpose.git
cd chairpose
conda env create -f environment.yml
conda activate chairpose


