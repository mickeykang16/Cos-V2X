"""
Model-only runtime benchmark (excludes data loading).

Usage:
    python scripts/benchmark_runtime.py \
        --config projects/configs/sparsedrive_small_stage2_6cams_v2x_top100.py \
        --checkpoint work_dirs/6cams_both_infra_v8_v2x_stage2_top100_fix/latest.pth \
        --warmup 10 \
        --iters 50
"""

import argparse
import os
import sys
import numpy as np
import torch

# ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# mmdet / mmcv imports
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

# SparseDrive custom dataset builder (handles temporal/sequence datasets)
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model inference time only")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=10,  help="# warmup iters (not counted)")
    parser.add_argument("--iters",  type=int, default=50,  help="# measured iters")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def move_to_device(data, device):
    """Recursively move tensors/DataContainers to device."""
    import mmcv
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, mmcv.parallel.DataContainer):
        return move_to_device(data.data, device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [move_to_device(v, device) for v in data]
    return data


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ── 1. Build config & dataset ──────────────────────────────────────────
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    # Force test dataset to samples_per_gpu=1
    # Remove samples_per_gpu from dataset cfg if present (it's a dataloader arg)
    test_cfg = cfg.data.test.copy()
    test_cfg.pop("samples_per_gpu", None)
    test_cfg.test_mode = True

    dataset = build_dataset(test_cfg)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
    )

    # ── 2. Build & load model ──────────────────────────────────────────────
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.to(device)
    model.eval()

    print(f"\n[benchmark] Config  : {args.config}")
    print(f"[benchmark] Ckpt    : {args.checkpoint}")
    print(f"[benchmark] Warmup  : {args.warmup}  |  Measured iters: {args.iters}")
    print(f"[benchmark] Device  : {device}\n")

    # ── 3. Pre-fetch ONE batch onto GPU (data loading excluded) ───────────
    data_iter = iter(dataloader)
    raw_batch = next(data_iter)

    # Unwrap DataContainers and move everything to GPU
    batch = move_to_device(raw_batch, device)

    # After move_to_device, DataContainer is unwrapped to its inner list/tensor.
    # img.data was [tensor(1, 6, 3, 256, 704)], so batch['img'] is now that list.
    img = batch.pop("img")
    if isinstance(img, (list, tuple)):
        img = img[0]  # tensor: [1, N_cams, C, H, W]  — keeps the batch dim

    # kwargs: other DataContainers are similarly unwrapped to lists; take [0]
    kwargs = {}
    for k, v in batch.items():
        kwargs[k] = v[0] if isinstance(v, (list, tuple)) else v

    # ── 4. Warmup ────────────────────────────────────────────────────────
    print("[benchmark] Running warmup ...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.simple_test(img, **kwargs)
    torch.cuda.synchronize()
    print("[benchmark] Warmup done.\n")

    # ── 5. Measure with CUDA events (accurate GPU timing) ─────────────────
    times = []
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i in range(args.iters):
            starter.record()
            _ = model.simple_test(img, **kwargs)
            ender.record()
            torch.cuda.synchronize()
            elapsed = starter.elapsed_time(ender)   # ms
            times.append(elapsed)
            if (i + 1) % 10 == 0:
                print(f"  iter {i+1:3d}/{args.iters}  |  {elapsed:.1f} ms")

    times = np.array(times)
    print("\n" + "=" * 50)
    print(f"  Mean   : {times.mean():.2f} ms  ({1000/times.mean():.2f} FPS)")
    print(f"  Median : {np.median(times):.2f} ms")
    print(f"  Std    : {times.std():.2f} ms")
    print(f"  Min    : {times.min():.2f} ms")
    print(f"  Max    : {times.max():.2f} ms")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
