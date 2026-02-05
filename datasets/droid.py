import pandas as pd
from tqdm import tqdm
from pathlib import Path
import decord
import shutil
import subprocess
import json
from typing import Dict, Any
from .video_base import VideoDataset, RGBXYZDataset


class DroidVideoDataset(VideoDataset):
    def __init__(self, cfg: Dict[str, Any], split: str = "training"):
        self.override_fps = cfg.download.override_fps
        self.views = cfg.download.views
        super().__init__(cfg, split)

    def download(self):
        self.data_root.mkdir(parents=True, exist_ok=True)

        # print("Downloading DROID dataset...")
        # cmd = f"gsutil -m cp -r gs://gresearch/robotics/droid_raw {self.data_root}"
        # subprocess.run(cmd, shell=True, check=True)
        # print("Download complete!")

        # build metadata
        raw_dir = self.data_root / "droid_raw"
        caption_file = raw_dir / "1.0.1" / "aggregated-annotations-030724.json"
        caption_data = json.load(open(caption_file))
        records = []
        for lab_dir in (raw_dir / "1.0.1").glob("*/"):
            print("processing", lab_dir)
            print("=" * 100)
            # Delete failure directory and its contents if it exists
            failure_dir = lab_dir / "failure"
            success_dir = lab_dir / "success"
            if failure_dir.exists():
                shutil.rmtree(failure_dir)

            for date_dir in list(success_dir.glob("*")):
                for episode_dir in list(date_dir.glob("*")):
                    # Rename episode directory if it contains ":"
                    if ":" in episode_dir.name:
                        new_name = episode_dir.name.replace(":", "_")
                        new_path = episode_dir.parent / new_name
                        if new_path.exists():
                            shutil.rmtree(episode_dir)
                        else:
                            episode_dir.rename(new_path)

            for episode_dir in tqdm(list(success_dir.glob("*/*"))):
                annotation_file = list(episode_dir.glob("*.json"))
                if not annotation_file:
                    continue
                annotation_file = annotation_file[0]
                f = json.load(open(annotation_file))
                caption = f["current_task"]
                uuid = f["uuid"]
                for views in self.views:
                    video_path = lab_dir / f[views + "_mp4_path"].replace(":", "_")
                    state_path = lab_dir / f["hdf5_path"].replace(":", "_")
                    n_frames = f["trajectory_length"]

                    if not video_path.exists():
                        print(f"Video file not found: {video_path}")
                        continue

                    try:
                        vr = decord.VideoReader(str(video_path))
                        fps = self.override_fps
                        width = 1280  # vr[0].shape[1]
                        height = 720  # vr[0].shape[0]

                        del vr
                    except Exception as e:
                        print(f"Error loading video {video_path}: {e}")
                        continue

                    video_path = video_path.relative_to(self.data_root)
                    # state_path = state_path.relative_to(self.data_root)

                    if uuid not in caption_data:
                        caption = ""
                        has_caption = False
                    else:
                        caption = caption_data[uuid]
                        has_caption = True
                    records.append(
                        {
                            "video_path": str(video_path),
                            # "state_path": str(state_path),
                            "original_caption": caption,
                            "fps": fps,
                            "n_frames": n_frames,
                            "width": width,
                            "height": height,
                            "has_caption": has_caption,
                        }
                    )
        metadata_path = self.data_root / self.metadata_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        df.to_csv(metadata_path, index=False)
        print(f"Created metadata CSV with {len(records)} videos")


class DroidRGBXYZDataset(RGBXYZDataset):
    def __init__(self, cfg: Dict[str, Any], split: str = "training"):
        self.override_fps = cfg.download.override_fps
        self.views = cfg.download.views
        super().__init__(cfg, split)

    def download(self):
        self.data_root.mkdir(parents=True, exist_ok=True)

        # print("Downloading DROID dataset...")
        # cmd = f"gsutil -m cp -r gs://gresearch/robotics/droid_raw {self.data_root}"
        # subprocess.run(cmd, shell=True, check=True)
        # print("Download complete!")

        # build metadata
        raw_dir = self.data_root / "droid_raw"
        caption_file = raw_dir / "1.0.1" / "aggregated-annotations-030724.json"
        caption_data = json.load(open(caption_file))
        records = []
        for lab_dir in (raw_dir / "1.0.1").glob("*/"):
            print("processing", lab_dir)
            print("=" * 100)
            # Delete failure directory and its contents if it exists
            failure_dir = lab_dir / "failure"
            success_dir = lab_dir / "success"
            if failure_dir.exists():
                shutil.rmtree(failure_dir)

            for date_dir in list(success_dir.glob("*")):
                for episode_dir in list(date_dir.glob("*")):
                    # Rename episode directory if it contains ":"
                    if ":" in episode_dir.name:
                        new_name = episode_dir.name.replace(":", "_")
                        new_path = episode_dir.parent / new_name
                        if new_path.exists():
                            shutil.rmtree(episode_dir)
                        else:
                            episode_dir.rename(new_path)

            for episode_dir in tqdm(list(success_dir.glob("*/*"))):
                annotation_file = list(episode_dir.glob("*.json"))
                if not annotation_file:
                    continue
                annotation_file = annotation_file[0]
                f = json.load(open(annotation_file))
                caption = f["current_task"]
                uuid = f["uuid"]
                for views in self.views:
                    video_path = lab_dir / f[views + "_mp4_path"].replace(":", "_")
                    state_path = lab_dir / f["hdf5_path"].replace(":", "_")
                    n_frames = f["trajectory_length"]

                    if not video_path.exists():
                        print(f"Video file not found: {video_path}")
                        continue

                    try:
                        vr = decord.VideoReader(str(video_path))
                        fps = self.override_fps
                        width = 1280  # vr[0].shape[1]
                        height = 720  # vr[0].shape[0]

                        del vr
                    except Exception as e:
                        print(f"Error loading video {video_path}: {e}")
                        continue

                    video_path = video_path.relative_to(self.data_root)
                    # state_path = state_path.relative_to(self.data_root)

                    if uuid not in caption_data:
                        caption = ""
                        has_caption = False
                    else:
                        caption = caption_data[uuid]
                        has_caption = True
                    records.append(
                        {
                            "video_path": str(video_path),
                            # "state_path": str(state_path),
                            "original_caption": caption,
                            "fps": fps,
                            "n_frames": n_frames,
                            "width": width,
                            "height": height,
                            "has_caption": has_caption,
                        }
                    )
        metadata_path = self.data_root / self.metadata_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        df.to_csv(metadata_path, index=False)
        print(f"Created metadata CSV with {len(records)} videos")