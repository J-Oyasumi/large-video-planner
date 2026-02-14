"""
Calculate metrics for video model inference result.

input_dir/
    pred_rgb_{batch_idx}_{idx}.mp4
    gt_rgb_{batch_idx}_{idx}.mp4
    pred_xyz_{batch_idx}_{idx}.npz (Optional)
    gt_xyz_{batch_idx}_{idx}.npz (Optional)

Gather all predicted and ground truth rgb video and calcute four metrics.
Processes in batches to avoid OOM from stacking all videos at once;
final metrics are averaged over batches.
"""
import argparse
from pathlib import Path
import json
from tqdm import tqdm
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


def _extract_metric_value(v):
    """Extract scalar from metric result (handles both plain scalar and {'value': [...]} format)."""
    if isinstance(v, dict) and "value" in v:
        vals = v["value"]
        return float(vals[0]) if len(vals) == 1 else float(np.mean(vals))
    return float(v)


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
    pred_paths = sorted(input_dir.glob("pred_rgb*.mp4"))
    for pred_rgb_path in tqdm(pred_paths, desc="Loading videos", unit="video"):
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

    batch_size = getattr(args, "batch_size", 4)
    rgb_metrics = calculate_rgb_metrics_batched(
        pred_rgb_list, gt_rgb_list, device=device, batch_size=batch_size
    )
    print("RGB metrics:")
    for k, v in rgb_metrics.items():
        print(f"  {k}: {v}")

    results = {"input_dir": str(input_dir), "rgb_metrics": _to_json_serializable(rgb_metrics)}

    if args.calculate_xyz and len(pred_xyz_list) > 0:
        xyz_metrics = calculate_xyz_metrics_batched(
            pred_xyz_list, gt_xyz_list, batch_size=batch_size
        )
        print("XYZ metrics:")
        for k, v in xyz_metrics.items():
            print(f"  {k}: {v}")
        results["xyz_metrics"] = _to_json_serializable(xyz_metrics)

    output_path = Path(args.output) if args.output else input_dir / "metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


@torch.no_grad()
def calculate_rgb_metrics_batched(
    pred_list: list, gt_list: list, device: str = "cuda", batch_size: int = 4, only_final: bool = True
):
    """Calculate RGB metrics in batches to avoid OOM; returns mean over all samples."""
    results_acc = {"fvd": 0.0, "ssim": 0.0, "psnr": 0.0, "lpips": 0.0}
    total = 0
    n = len(pred_list)
    pbar = tqdm(total=n, desc="RGB metrics", unit="sample")
    for i in range(0, n, batch_size):
        batch_pred = torch.stack(pred_list[i : i + batch_size], dim=0).to(device)
        batch_gt = torch.stack(gt_list[i : i + batch_size], dim=0).to(device)
        b = batch_pred.shape[0]
        m = calculate_rgb_metrics(batch_pred, batch_gt, device=device, only_final=only_final)
        for k in results_acc:
            val = _extract_metric_value(m[k])
            results_acc[k] += val * b
        total += b
        postfix = {k: f"{v / total:.4f}" for k, v in results_acc.items()}
        pbar.update(b)
        pbar.set_postfix(postfix)
        del batch_pred, batch_gt
        if device == "cuda":
            torch.cuda.empty_cache()
    pbar.close()
    return {k: v / total for k, v in results_acc.items()}


@torch.no_grad()
def calculate_rgb_metrics(rgb_1: torch.Tensor, rgb_2: torch.Tensor, device: str = "cuda", only_final: bool = True):
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
def calculate_xyz_metrics_batched(pred_list: list, gt_list: list, batch_size: int = 4):
    """Calculate XYZ metrics in batches to avoid OOM; returns aggregated stats over all samples."""
    cd_sum = 0.0
    cd_sum_sq = 0.0
    cd_min = float("inf")
    cd_max = float("-inf")
    total_count = 0
    n = len(pred_list)
    chamfer_fn = ChamferDistance()
    pbar = tqdm(total=n, desc="XYZ metrics", unit="sample")
    for i in range(0, n, batch_size):
        batch_pred = torch.stack(pred_list[i : i + batch_size], dim=0)
        batch_gt = torch.stack(gt_list[i : i + batch_size], dim=0)
        xyz_1 = rearrange(batch_pred, "b t h w c -> (b t) (h w) c")
        xyz_2 = rearrange(batch_gt, "b t h w c -> (b t) (h w) c")
        cd = chamfer_fn(xyz_1, xyz_2, batch_reduction=None, point_reduction="mean", bidirectional=True)
        cd = cd.cpu().numpy()
        count = cd.size
        cd_sum += cd.sum()
        cd_sum_sq += (cd ** 2).sum()
        cd_min = min(cd_min, cd.min())
        cd_max = max(cd_max, cd.max())
        total_count += count
        mean_cur = cd_sum / total_count
        pbar.update(len(pred_list[i : i + batch_size]))
        pbar.set_postfix(mean=f"{mean_cur:.4f}", min=f"{cd_min:.4f}", max=f"{cd_max:.4f}")
        del batch_pred, batch_gt, xyz_1, xyz_2, cd
    pbar.close()
    mean = cd_sum / total_count
    var = cd_sum_sq / total_count - mean ** 2
    std = np.sqrt(max(var, 0.0))
    return {
        "chamfer_distance": dict(mean=float(mean), std=float(std), min=float(cd_min), max=float(cd_max))
    }


@torch.no_grad()
def calculate_xyz_metrics(xyz_1: torch.Tensor, xyz_2: torch.Tensor):
    """
    Calculate XYZ metrics between two batch of pointclouds.
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
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing to avoid OOM")
    args = parser.parse_args()
    main(args)