#!/usr/bin/env python3
"""
Standalone inference script -- the judged, distributable deliverable.

Loads the trained proposed model ONCE, then processes every degraded image
in --input_dir, restoring it and saving the result to --output_dir with the
same filename. No notebook, no training-only dependencies (no lpips,
no tensorboard) -- lightweight and safe to run on an isolated/offline GPU
box (e.g. the judges' H100 environment).

USAGE:
    python inference.py --input_dir /path/to/degraded --output_dir /path/to/restored
    python inference.py --input_dir X --output_dir Y --checkpoint path/to/model.pth --batch_size 16 --half

Requires only: torch, numpy (both already needed by the rest of this repo).
"""
import argparse
import os
import sys
import time
import glob

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.restoration_net import DistributionMixtureRestorationNet


def list_input_files(input_dir):
    paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    paths = [p for p in paths if "__MACOSX" not in p and not os.path.basename(p).startswith("._")]
    return paths


def load_model(checkpoint_path, device):
    model = DistributionMixtureRestorationNet(base_ch=32, n_components=3,
                                                n_lr_blocks=4, n_hr_blocks=2, use_film=True)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, ckpt.get("epoch", "unknown")


def print_environment_info(device):
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
        print(f"cuDNN version:   {torch.backends.cudnn.version()}")
    print(f"Device selected: {device}")
    print("=" * 60)


@torch.no_grad()
def run_batch(model, batch_np, device, use_half):
    """batch_np: numpy array [B, H, W]. Returns restored numpy array [B, H2, W2]."""
    x = torch.from_numpy(batch_np).unsqueeze(1).to(device, non_blocking=True)  # [B,1,H,W]
    if use_half and device == "cuda":
        x = x.half()
    with torch.autocast(device_type=device, enabled=(device == "cuda")):
        restored, _, _, _ = model(x)
    restored = restored.float().clamp(0, 1).cpu().numpy()[:, 0]  # [B,H2,W2]
    return restored


def main():
    parser = argparse.ArgumentParser(description="Restore degraded semiconductor inspection images.")
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Directory containing degraded .npy input images.")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Directory to write restored .npy output images (same filenames).")
    parser.add_argument("--checkpoint", type=str,
                         default="results/checkpoints/DistributionMixtureRestorationNet.pth",
                         help="Path to the trained proposed-model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=16,
                         help="Images per GPU batch. All inputs are fixed-size (128x128), "
                              "so batching is straightforward and improves throughput.")
    parser.add_argument("--half", action="store_true",
                         help="Use FP16 inference for higher throughput on modern GPUs (e.g. H100).")
    parser.add_argument("--device", type=str, default=None,
                         help="Force a device ('cuda' or 'cpu'). Default: auto-detect.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print_environment_info(device)

    if device == "cuda":
        # Fixed input size (128x128) across the whole dataset -> cudnn.benchmark
        # lets cuDNN pick the fastest convolution algorithm and cache it,
        # a straightforward, safe throughput win on any NVIDIA GPU including H100.
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading model from: {args.checkpoint}")
    model, ckpt_epoch = load_model(args.checkpoint, device)
    if args.half and device == "cuda":
        model = model.half()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded (checkpoint epoch: {ckpt_epoch}, {n_params:,} parameters).")

    input_paths = list_input_files(args.input_dir)
    if len(input_paths) == 0:
        print(f"No .npy files found in {args.input_dir} -- nothing to do.")
        return
    print(f"\nFound {len(input_paths)} input images in {args.input_dir}")
    print(f"Batch size: {args.batch_size}  |  FP16: {args.half}\n")

    n_processed = 0
    n_failed = 0
    per_image_times = []
    t_start = time.time()

    for batch_start in range(0, len(input_paths), args.batch_size):
        batch_paths = input_paths[batch_start: batch_start + args.batch_size]
        batch_arrays, valid_paths = [], []
        for p in batch_paths:
            try:
                arr = np.load(p).astype(np.float32)
                batch_arrays.append(arr)
                valid_paths.append(p)
            except Exception as e:
                print(f"  [SKIP] Failed to load {p}: {e}")
                n_failed += 1

        if not batch_arrays:
            continue

        batch_np = np.stack(batch_arrays, axis=0)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        try:
            restored_batch = run_batch(model, batch_np, device, args.half)
        except Exception as e:
            print(f"  [ERROR] Batch starting at {valid_paths[0]} failed: {e}")
            n_failed += len(valid_paths)
            continue
        if device == "cuda":
            torch.cuda.synchronize()
        batch_time = time.time() - t0
        per_image_times.extend([batch_time / len(valid_paths)] * len(valid_paths))

        for p, restored_np in zip(valid_paths, restored_batch):
            out_path = os.path.join(args.output_dir, os.path.basename(p))
            np.save(out_path, restored_np.astype(np.float32))
            n_processed += 1

        if n_processed % (args.batch_size * 5) < args.batch_size:
            print(f"  processed {n_processed}/{len(input_paths)}")

    total_time = time.time() - t_start

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Images processed:         {n_processed}")
    print(f"Images failed:            {n_failed}")
    print(f"Total wall-clock time:    {total_time:.2f} sec")
    if per_image_times:
        mean_ms = np.mean(per_image_times) * 1000
        throughput = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
        print(f"Mean per-image latency:  {mean_ms:.2f} ms")
        print(f"Throughput:              {throughput:.2f} images/sec")
    print(f"Output directory:        {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
