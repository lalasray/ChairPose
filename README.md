# ChairPose

**ChairPose** is a framework for estimating full-body 3D seated poses from pressure sensor data on arbitrary chairs, using a discrete motion representation and pressure-conditioned generative modeling.

---

## Features

- **MotionQuantizer**: Compresses 3D motion into discrete tokens via Vector Quantized VAE.
- **Pressure2Pose**: Autoregressive classifier predicting pose tokens from pressure and chair geometry.

---

## Setup

1. Clone repository and install dependencies:

```bash
git clone https://github.com/your-repo/chairpose.git
cd chairpose
conda env create -f environment.yml
conda activate chairpose


