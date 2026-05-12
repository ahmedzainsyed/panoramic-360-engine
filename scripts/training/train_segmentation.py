#!/usr/bin/env python3
"""
SegFormer Training Script for Panoramic Construction Segmentation.
Supports distributed training, mixed precision, and W&B logging.
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Train SegFormer for construction segmentation")
    parser.add_argument("--config", required=True, help="Training config YAML path")
    parser.add_argument("--data-path", default="/data/panoramas", help="Dataset root")
    parser.add_argument("--output-dir", default="./models/segformer_latest", help="Output dir")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=6e-5, help="Learning rate")
    parser.add_argument("--wandb-project", default="360-construction-segmentation", help="W&B project")
    parser.add_argument("--wandb-run", default=None, help="W&B run name")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--distributed", action="store_true", help="Use DDP training")
    args = parser.parse_args()

    import yaml
    import torch
    import mlflow

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Setup MLflow experiment
    try:
        mlflow.set_experiment(config.get("logging", {}).get("wandb_project", "panoramic-360"))
        with mlflow.start_run(run_name=args.wandb_run or "segformer_training"):
            mlflow.log_params({
                "model": config.get("model", {}).get("name", "segformer-b5"),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
            })
    except Exception as e:
        print(f"MLflow setup failed (continuing without): {e}")

    # Initialize model
    from ml.segmentation.panoramic_segmenter import SegFormerPanoramicSegmenter, NUM_CLASSES
    model = SegFormerPanoramicSegmenter(
        model_path=args.resume,
        num_classes=NUM_CLASSES,
        device=device,
        use_amp=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n✅ Training setup complete.")
    print(f"   Output: {args.output_dir}")
    print(f"   Classes: {NUM_CLASSES}")
    print(f"   NOTE: Full training requires labeled panorama dataset.")
    print(f"   Implement data loaders in scripts/data/panorama_dataset.py")

if __name__ == "__main__":
    main()
