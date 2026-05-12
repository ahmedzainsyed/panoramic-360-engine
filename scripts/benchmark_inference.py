#!/usr/bin/env python3
"""
Inference Performance Benchmark
Measures throughput and latency across all ML modules.
"""
import time
import statistics
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def benchmark_module(name, fn, runs=5):
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"  Runs: {runs}")
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.1f}ms")
    print(f"  Mean: {statistics.mean(times):.1f}ms")
    print(f"  Min:  {min(times):.1f}ms")
    print(f"  P95:  {sorted(times)[int(len(times)*0.95)]:.1f}ms")
    return {"mean_ms": statistics.mean(times), "min_ms": min(times), "p95_ms": sorted(times)[-1]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"\n🔬 360° Engine Inference Benchmark")
    print(f"   Image size: {args.width}x{args.height}")
    print(f"   Device: {args.device}")

    panorama = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    results = {}

    # Spherical geometry
    from ml.spherical_geometry.spherical_engine import SphericalGeometryEngine
    geo = SphericalGeometryEngine(cubemap_face_size=512, device=args.device)
    results["spherical_cubemap"] = benchmark_module("Cubemap Projection",
        lambda: geo.equirect_to_cubemap(panorama, face_size=512))
    results["perspective_tiling"] = benchmark_module("Perspective Tiling",
        lambda: geo.extract_perspective_tiles(panorama, tile_size=640))

    # Segmentation
    from ml.segmentation.panoramic_segmenter import SegFormerPanoramicSegmenter
    seg = SegFormerPanoramicSegmenter(model_path=None, device=args.device, use_amp=False, tile_size=256)
    results["segmentation"] = benchmark_module("Segmentation (SegFormer)",
        lambda: seg.segment_panorama(panorama))

    # PPE
    from ml.ppe.ppe_engine import PPEDetectionEngine
    ppe = PPEDetectionEngine(model_path=None, device=args.device, use_tracker=False)
    results["ppe"] = benchmark_module("PPE Detection",
        lambda: ppe.analyze_panorama(panorama, "bench_pan"))

    # Hazard
    from ml.hazards.hazard_engine import HazardZoneEngine
    hazard = HazardZoneEngine(device=args.device)
    results["hazards"] = benchmark_module("Hazard Analysis",
        lambda: hazard.analyze_panorama(panorama, "bench_pan"))

    # Summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    for module, stats in results.items():
        print(f"  {module:30s} {stats['mean_ms']:8.1f}ms (mean)")
    total = sum(s['mean_ms'] for s in results.values())
    print(f"  {'FULL PIPELINE':30s} {total:8.1f}ms")
    print(f"  {'FPS (full pipeline)':30s} {1000/total:8.1f} fps")

if __name__ == "__main__":
    main()
