# GPU Setup Guide

## Prerequisites
- NVIDIA GPU with CUDA 12.1+ (RTX 3080+ or A-series recommended)
- NVIDIA Driver 525+
- Docker 24+

## Step 1: Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release && echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Step 2: Verify GPU Access in Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx   Driver Version: 525.xx.xx   CUDA Version: 12.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX ...     Off  | 00000000:01:00.0 Off |                  N/A |
```

## Step 3: Start with GPU Support

```bash
make prod-gpu
# or:
docker compose -f docker/docker-compose.gpu.yml up -d
```

## Step 4: Verify GPU in Container

```bash
make shell
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## TensorRT Optimization (Optional)

For maximum inference performance, export models to TensorRT:

```bash
# Run inside GPU container
make export-tensorrt

# This exports:
#   - SegFormer → segformer_b5_fp16.engine
#   - YOLOv8 PPE → yolo_ppe_fp16.engine
# Typical speedup: 2-4x over standard PyTorch
```

## Multi-GPU Setup

Edit `configs/inference/inference_config.yaml`:
```yaml
inference:
  device: cuda
  gpu_ids: [0, 1]  # Use both GPUs
```

Configure Kubernetes for multi-GPU:
```yaml
resources:
  limits:
    nvidia.com/gpu: 2
```

## Memory Guidelines

| GPU VRAM | Recommended Config |
|----------|-------------------|
| 8GB  | batch_size=1, tile_size=512 |
| 16GB | batch_size=2, tile_size=1024 |
| 24GB | batch_size=4, tile_size=1024, TTA enabled |
| 40GB+ | batch_size=8, full pipeline parallel |
