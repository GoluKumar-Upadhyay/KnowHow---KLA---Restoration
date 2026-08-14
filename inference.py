
import argparse
import os
import sys
import time
import glob

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.restoration_net import DistributionMixtureRestorationNet

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_input_files(input_dir):
    paths = []
    for ext in (".npy",) + IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    paths = sorted(set(paths))
    paths = [p for p in paths if "__MACOSX" not in p and not os.path.basename(p).startswith("._")]
    return paths


def load_input_array(path, target_size=128):
    """Loads .npy directly (preserving true values, which may legitimately
    exceed [0,1] -- see Section IV of the paper); loads any other supported
    image format via PIL, converts to grayscale, and resizes to target_size
    if not already that size."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr
    else:
        img = Image.open(path).convert("L")
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.BILINEAR)
        return np.array(img).astype(np.float32) / 255.0


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


def save_comparison_grid(pairs, out_path):
    """pairs: list of (filename, degraded_np, restored_np). Saves one combined
    PNG with a [Noisy | Restored] row per pair, labeled by filename -- matches
    the visual style already used in Proposed_Model_Testing.ipynb."""
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for row, (name, degraded, restored) in enumerate(pairs):
        axes[row, 0].imshow(degraded, cmap="gray")
        axes[row, 0].set_title(f"Noisy: {name}", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(restored, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Restored: {name}", fontsize=9)
        axes[row, 1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Restore degraded semiconductor inspection images.")
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Directory containing degraded input images "
                              "(.npy, .png, .jpg, .jpeg, .bmp, .tif, .tiff).")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Directory to write restored outputs.")
    parser.add_argument("--checkpoint", type=str,
                         default="results/checkpoints/DistributionMixtureRestorationNet.pth",
                         help="Path to the trained proposed-model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=16,
                         help="Images per GPU batch. All inputs are resized to a fixed 128x128, "
                              "so batching is straightforward and improves throughput.")
    parser.add_argument("--half", action="store_true",
                         help="Use FP16 inference for higher throughput on modern GPUs (e.g. H100). "
                              "Ignored automatically on CPU (no benefit, some ops unsupported).")
    parser.add_argument("--device", type=str, default=None,
                         help="Force a device ('cuda' or 'cpu'). Default: auto-detect -- "
                              "runs correctly on either with no other changes needed.")
    parser.add_argument("--grid_size", type=int, default=20,
                         help="Number of [Noisy | Restored] pairs per combined comparison PNG. "
                              "If the total image count fits in one grid_size, only one file is produced.")
    parser.add_argument("--no_grid", action="store_true",
                         help="Skip combined comparison PNG generation; save only .npy restored arrays.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print_environment_info(device)

    use_half = args.half
    if use_half and device == "cpu":
        print("NOTE: --half requested but device is CPU -- FP16 gives no benefit and some ops are "
              "unsupported on CPU. Ignoring --half for this run.\n")
        use_half = False

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    npy_dir = os.path.join(args.output_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading model from: {args.checkpoint}")
    model, ckpt_epoch = load_model(args.checkpoint, device)
    if use_half:
        model = model.half()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded (checkpoint epoch: {ckpt_epoch}, {n_params:,} parameters).")

    input_paths = list_input_files(args.input_dir)
    if len(input_paths) == 0:
        print(f"No supported input files found in {args.input_dir} "
              f"(looked for .npy, {', '.join(IMAGE_EXTS)}) -- nothing to do.")
        return
    print(f"\nFound {len(input_paths)} input images in {args.input_dir}")
    print(f"Batch size: {args.batch_size}  |  FP16: {use_half}  |  Combined grid PNG: {not args.no_grid}\n")

    n_processed = 0
    n_failed = 0
    per_image_times = []
    grid_buffer = []
    grid_file_idx = 1
    t_start = time.time()

    for batch_start in range(0, len(input_paths), args.batch_size):
        batch_paths = input_paths[batch_start: batch_start + args.batch_size]
        batch_arrays, valid_paths = [], []
        for p in batch_paths:
            try:
                arr = load_input_array(p)
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
            restored_batch = run_batch(model, batch_np, device, use_half)
        except Exception as e:
            print(f"  [ERROR] Batch starting at {valid_paths[0]} failed: {e}")
            n_failed += len(valid_paths)
            continue
        if device == "cuda":
            torch.cuda.synchronize()
        batch_time = time.time() - t0
        per_image_times.extend([batch_time / len(valid_paths)] * len(valid_paths))

        for p, degraded_np, restored_np in zip(valid_paths, batch_np, restored_batch):
            base = os.path.splitext(os.path.basename(p))[0]
            np.save(os.path.join(npy_dir, base + ".npy"), restored_np.astype(np.float32))

            if not args.no_grid:
                grid_buffer.append((os.path.basename(p), degraded_np, restored_np))
                if len(grid_buffer) >= args.grid_size:
                    out_path = os.path.join(args.output_dir, f"comparison_grid_{grid_file_idx:04d}.png")
                    save_comparison_grid(grid_buffer, out_path)
                    print(f"  saved {out_path} ({len(grid_buffer)} images)")
                    grid_buffer = []
                    grid_file_idx += 1

            n_processed += 1

        if n_processed % (args.batch_size * 5) < args.batch_size:
            print(f"  processed {n_processed}/{len(input_paths)}")

    if not args.no_grid and grid_buffer:
        out_path = os.path.join(args.output_dir, f"comparison_grid_{grid_file_idx:04d}.png")
        save_comparison_grid(grid_buffer, out_path)
        print(f"  saved {out_path} ({len(grid_buffer)} images)")

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
    print(f"Restored .npy:            {os.path.abspath(npy_dir)}")
    if not args.no_grid:
        print(f"Comparison grid PNG(s):   {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()