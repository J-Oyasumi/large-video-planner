"""
Calculate metrics for video model inference result.

input_dir/
    pred_rgb_{batch_idx}_{frame_idx}.mp4
    gt_rgb_{batch_idx}_{frame_idx}.mp4
    pred_xyz_{batch_idx}_{frame_idx}.npz (Optional)
    gt_xyz_{batch_idx}_{frame_idx}.npz (Optional)

Gather all predicted and ground truth rgb video and calcute four metrics.
All predictions are stacked into a single batch, and all ground truths
are stacked into a single batch, then metrics are computed on the
two batches.
"""
import argparse
from pathlib import Path
import json

import decord
import numpy as np
import torch
from chamferdist import ChamferDistance
from einops import rearrange

import video_metric.calculate_fvd as FVD
import video_metric.calculate_lpips as LPIPS
import video_metric.calculate_psnr as PSNR
import video_metric.calculate_ssim as SSIM


def _load_rgb_video(path: Path) -> torch.Tensor:
    """Load an RGB video from an mp4 file.

    Returns:
        Tensor of shape (T, C, H, W), dtype=float32, range [0, 1].
    """
    vr = decord.VideoReader(str(path))
    if len(vr) == 0:
        raise ValueError(f"Empty video: {path}")
    frames = vr.get_batch(range(len(vr)))  # (T, H, W, 3), uint8
    frames = frames.asnumpy().astype("float32") / 255.0
    frames = torch.from_numpy(frames)  # (T, H, W, 3)
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
    return frames


def _to_json_serializable(obj):
    """Convert torch/numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "item"):  # torch.Tensor
        return obj.item()
    return obj


def _load_xyz(path: Path) -> torch.Tensor:
    """Load XYZ npz file as torch tensor.
    Returns:
        Tensor of shape (T, H, W, 3)
    """
    data = np.load(str(path))["xyz"]
    return torch.from_numpy(data.astype("float32"))


def main(args):
    input_dir = Path(args.input_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pred_rgb_list = []
    gt_rgb_list = []
    pred_xyz_list = []
    gt_xyz_list = []

    # Collect all pred_rgb files and their corresponding gt_rgb (and xyz if present).
    for pred_rgb_path in sorted(input_dir.glob("pred_rgb_*.mp4")):
        suffix = pred_rgb_path.name[len("pred_rgb_") :]  # "{batch_idx}_{frame_idx}.mp4"
        gt_rgb_path = input_dir / f"gt_rgb_{suffix}"

        if not gt_rgb_path.exists():
            # Skip if no matching GT.
            continue

        pred_rgb = _load_rgb_video(pred_rgb_path)
        gt_rgb = _load_rgb_video(gt_rgb_path)

        pred_rgb_list.append(pred_rgb)
        gt_rgb_list.append(gt_rgb)

        if args.calculate_xyz:
            pred_xyz_path = input_dir / f"pred_xyz_{suffix.replace('.mp4', '.npz')}"
            gt_xyz_path = input_dir / f"gt_xyz_{suffix.replace('.mp4', '.npz')}"
            if pred_xyz_path.exists() and gt_xyz_path.exists():
                pred_xyz_list.append(_load_xyz(pred_xyz_path))
                gt_xyz_list.append(_load_xyz(gt_xyz_path))

    if len(pred_rgb_list) == 0:
        raise RuntimeError(f"No valid pred/gt RGB pairs found in {input_dir}")

    # Stack all videos into a single batch: (B, T, C, H, W)
    pred_rgb_batch = torch.stack(pred_rgb_list, dim=0).to(device)
    gt_rgb_batch = torch.stack(gt_rgb_list, dim=0).to(device)

    rgb_metrics = calculate_rgb_metrics(pred_rgb_batch, gt_rgb_batch, device=device)
    print("RGB metrics:")
    for k, v in rgb_metrics.items():
        print(f"  {k}: {v}")

    results = {"input_dir": str(input_dir), "rgb_metrics": _to_json_serializable(rgb_metrics)}

    if args.calculate_xyz and len(pred_xyz_list) > 0:
        pred_xyz_batch = torch.stack(pred_xyz_list, dim=0)
        gt_xyz_batch = torch.stack(gt_xyz_list, dim=0)

        xyz_metrics = calculate_xyz_metrics(pred_xyz_batch, gt_xyz_batch)
        print("XYZ metrics:")
        for k, v in xyz_metrics.items():
            print(f"  {k}: {v}")
        results["xyz_metrics"] = _to_json_serializable(xyz_metrics)

    output_path = Path(args.output) if args.output else input_dir / "metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {output_path}")

@torch.no_grad()
def calculate_rgb_metrics(rgb_1: torch.Tensor, rgb_2: torch.Tensor, device: str = "cuda", only_final: bool = False):
    """
    Calculate RGB metrics between two batch of videos.
    """
    results = {}
    results["fvd"] = FVD(rgb_1, rgb_2, device=device, method='styleganv', only_final=only_final)
    results["ssim"] = SSIM(rgb_1.cpu(), rgb_2.cpu(), only_final=only_final)
    results["psnr"] = PSNR(rgb_1.cpu(), rgb_2.cpu(), only_final=only_final)
    results["lpips"] = LPIPS(rgb_1, rgb_2, device=device, only_final=only_final)
    return results

@torch.no_grad()
def calculate_xyz_metrics(xyz_1: torch.Tensor, xyz_2: torch.Tensor):
    """
    Calculate XYZ metrics between two batch ofpointclouds.
    Args:
        xyz_1: (B, T, H, W, 3)
        xyz_2: (B, T, H, W, 3)
    Returns:
        results: Dictionary containing the metrics
            - "chamfer_distance": Dictionary containing the chamfer distance metrics
                - "mean": Mean chamfer distance
                - "std": Standard deviation of chamfer distance
                - "min": Minimum chamfer distance
                - "max": Maximum chamfer distance
    """
    results = {}
    xyz_1 = rearrange(xyz_1, "b t h w c -> (b t) (h w) c")
    xyz_2 = rearrange(xyz_2, "b t h w c -> (b t) (h w) c")
    chamfer_distance = ChamferDistance()(xyz_1, xyz_2, batch_reduction=None, point_reduction="mean", bidirectional=True)

    results["chamfer_distance"] = dict(
        mean=chamfer_distance.mean().item(),
        std=chamfer_distance.std().item(),
        min=chamfer_distance.min().item(), 
        max=chamfer_distance.max().item(),
    )

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path. Default: input_dir/metrics.json")
    parser.add_argument("--calculate_xyz", action="store_true")
    args = parser.parse_args()
    main(args)