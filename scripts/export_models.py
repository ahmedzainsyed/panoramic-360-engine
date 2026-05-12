#!/usr/bin/env python3
"""
Export models to ONNX / TensorRT for optimized production inference.
Usage:
  python scripts/export_models.py --format onnx
  python scripts/export_models.py --format tensorrt --precision fp16
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def export_segformer_onnx(model_path: str, output_path: str, input_h: int = 512, input_w: int = 1024):
    """Export SegFormer to ONNX with dynamic batch axis."""
    print(f"Exporting SegFormer: {model_path} → {output_path}")
    try:
        import torch
        from ml.segmentation.panoramic_segmenter import SegFormerPanoramicSegmenter
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SegFormerPanoramicSegmenter(model_path=model_path, device=device, use_amp=False)
        model.model.eval()
        dummy = torch.randn(1, 3, input_h, input_w).to(device)
        torch.onnx.export(
            model.model, dummy, output_path,
            opset_version=17,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
        )
        size_mb = os.path.getsize(output_path) / 1e6
        print(f"  ✓ Exported ONNX: {output_path} ({size_mb:.0f} MB)")
    except Exception as e:
        print(f"  ✗ Export failed: {e}")


def export_yolo_onnx(model_path: str, output_path: str):
    """Export YOLOv8 to ONNX."""
    print(f"Exporting YOLO: {model_path} → {output_path}")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.export(format="onnx", simplify=True, opset=17, dynamic=True, half=False)
        print(f"  ✓ Exported YOLO ONNX")
    except Exception as e:
        print(f"  ✗ YOLO export failed: {e}")


def convert_onnx_to_tensorrt(onnx_path: str, engine_path: str, precision: str = "fp16"):
    """Convert ONNX model to TensorRT engine."""
    print(f"Converting to TensorRT [{precision}]: {onnx_path} → {engine_path}")
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        with open(onnx_path, "rb") as f:
            parser.parse(f.read())
        engine_bytes = builder.build_serialized_network(network, config)
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        size_mb = os.path.getsize(engine_path) / 1e6
        print(f"  ✓ TensorRT engine: {engine_path} ({size_mb:.0f} MB)")
    except ImportError:
        print("  ✗ tensorrt package not available. Install TensorRT SDK.")
    except Exception as e:
        print(f"  ✗ TRT conversion failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export ML models for production inference")
    parser.add_argument("--format", choices=["onnx", "tensorrt", "both"], default="onnx")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--models-dir", default="ml/models")
    parser.add_argument("--output-dir", default="ml/models/exported")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = args.models_dir
    out_dir = args.output_dir

    print(f"\n🚀 Model Export Pipeline")
    print(f"   Format:    {args.format}")
    print(f"   Precision: {args.precision}")
    print(f"   Output:    {out_dir}\n")

    # SegFormer
    seg_path = os.path.join(models_dir, "segformer_b5_construction_v2.pth")
    if os.path.exists(seg_path):
        seg_onnx = os.path.join(out_dir, "segformer_b5_construction.onnx")
        export_segformer_onnx(seg_path, seg_onnx)
        if args.format in ("tensorrt", "both") and os.path.exists(seg_onnx):
            convert_onnx_to_tensorrt(seg_onnx, seg_onnx.replace(".onnx", f"_{args.precision}.engine"), args.precision)
    else:
        print(f"⚠️  SegFormer weights not found: {seg_path}")

    # PPE Model
    ppe_path = os.path.join(models_dir, "yolov8m_ppe_v2.pt")
    if os.path.exists(ppe_path):
        export_yolo_onnx(ppe_path, os.path.join(out_dir, "yolo_ppe_v2.onnx"))
    else:
        print(f"⚠️  PPE model not found: {ppe_path}")

    print(f"\n✅ Export complete. Check {out_dir}/")


if __name__ == "__main__":
    main()
