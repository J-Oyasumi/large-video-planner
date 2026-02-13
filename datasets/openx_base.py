from collections import defaultdict
import io

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from PIL import Image

from .video_base import VideoDataset
from utils.video_utils import write_numpy_to_mp4


class OpenXVideoDataset(VideoDataset):
    def preprocess_record(self, record):
        record["fps"] = self.cfg.download.openx_fps
        # if "bbox" in record:
        #     bbox = eval(record["bbox"])
        #     if len(bbox) == 5:
        #         record["has_bbox"] = True
        #         record["bbox_left"] = bbox[0]
        #         record["bbox_top"] = bbox[1]
        #         record["bbox_right"] = bbox[2]
        #         record["bbox_bottom"] = bbox[3]
        #     else:
        #         record["has_bbox"] = False
        #         record["bbox_left"] = 0
        #         record["bbox_top"] = 0
        #         record["bbox_right"] = 0
        #         record["bbox_bottom"] = 0
        return record

    def download(self):
        import tensorflow_datasets as tfds
        import tensorflow as tf
        from utils.tf_utils import recursive_cast_to_numpy

        all_episode_dir = self.data_root / "episodes"
        all_episode_dir.mkdir(parents=True, exist_ok=True)

        builder = tfds.builder_from_directory(
            builder_dir=f"gs://gresearch/robotics/{self.cfg.download.openx_name}/{self.cfg.download.openx_version}"
        )
        info = builder.info
        n_episodes = info.splits["train"].num_examples
        obs_feature = info.features["steps"].feature["observation"]
        image_obs_keys = [
            k for k, v in obs_feature.items() if isinstance(v, tfds.features.Image)
        ]
        # Disable tf.image.decode_image for observation images. We decode bytes with PIL
        # to avoid strict PNG decoder failures on malformed samples.
        decoders = {"steps": {"observation": {}}}
        for k in image_obs_keys:
            decoders["steps"]["observation"][k] = tfds.decode.SkipDecoding()

        # Count number of episodes to skip based on existing state files
        episode_id = 0
        for episode_id in range(n_episodes):
            episode_dir = all_episode_dir / f"episode_{episode_id}"
            state_path = episode_dir / "states.pkl"
            if not state_path.exists():
                break

        if episode_id > 0:
            print(f"Skipping {episode_id} already downloaded episodes")
        decode_errors = (
            tf.errors.InvalidArgumentError,
            tf.errors.DataLossError,
            tf.errors.OutOfRangeError,
        )

        pbar = tqdm(total=n_episodes - episode_id)
        while episode_id < n_episodes:
            episode_dir = all_episode_dir / f"episode_{episode_id}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            episode_records = defaultdict(list)
            state_path = episode_dir / "states.pkl"
            if state_path.exists():
                episode_id += 1
                pbar.update(1)
                continue
            try:
                # Read one episode at a time so a bad future sample doesn't crash
                # the whole streaming iterator due parallel decode/prefetch.
                dataset = builder.as_dataset(
                    split=f"train[{episode_id}:{episode_id + 1}]",
                    decoders=decoders,
                )
                episode_data = next(iter(dataset))
            except decode_errors as e:
                # Some OpenX shards contain malformed image payloads.
                # Skip this episode and continue from the next one.
                print(f"Skipping episode {episode_id} due to decode error: {e}")
                episode_id += 1
                pbar.update(1)
                continue

            try:
                episode = defaultdict(list)
                videos = defaultdict(list)
                fields_to_stack = []
                for k, v in episode_data.items():
                    if k != "steps":
                        episode[k] = recursive_cast_to_numpy(v)

                # sometimes we can split a video into multiple segments based on caption
                segments = {
                    "natural_language_instruction": [],
                    "instruction": [],
                    "language_instruction": [],
                    "language_instruction_2": [],
                    "language_instruction_3": [],
                }
                idx = -1
                # Per-step image decode can fail for a subset of frames in Open-X.
                # Ignore broken steps and keep valid ones from the same episode.
                try:
                    step_iter = episode_data["steps"].ignore_errors()
                except AttributeError:
                    step_iter = episode_data["steps"].apply(
                        tf.data.experimental.ignore_errors()
                    )

                for idx, step in enumerate(step_iter):
                    step = recursive_cast_to_numpy(step)
                    obs_dict = step["observation"]
                    action_dict = step["action"]
                    if hasattr(obs_dict, "shape"):
                        obs_dict = dict(observation=obs_dict)
                    if hasattr(action_dict, "shape"):
                        action_dict = dict(action=action_dict)

                    # some times caption field is here but mostly in observation
                    for k, v in step.items():
                        if k in segments:
                            obs_dict[k] = v

                    for k, v in obs_dict.items():
                        if k in image_obs_keys and isinstance(v, (bytes, bytearray)):
                            try:
                                img = Image.open(io.BytesIO(v)).convert("RGB")
                                videos[k].append(np.array(img))
                                continue
                            except Exception as e:
                                raise tf.errors.InvalidArgumentError(
                                    node_def=None,
                                    op=None,
                                    message=f"Failed to decode image bytes for key '{k}': {e}",
                                )
                        if (
                            hasattr(v, "shape")
                            and len(v.shape) == 3
                            and v.shape[-1] == 3
                        ):
                            videos[k].append(v)
                        elif k in segments:
                            if (
                                k == "instruction"
                                and self.cfg.download.openx_name == "language_table"
                            ):
                                # special case for language table dataset
                                v = tf.convert_to_tensor(v)
                                v = tf.strings.unicode_encode(v, output_encoding="UTF-8")
                                v = v.numpy().decode("utf-8").split("\x00")[0]
                            if not segments[k] or segments[k][-1][1] != v:
                                segments[k].append((idx, v))
                        elif k != "natural_language_embedding":
                            if hasattr(v, "shape"):
                                fields_to_stack.append("observation/" + k)
                            episode["observation/" + k].append(v)

                    for k, v in action_dict.items():
                        fields_to_stack.append("action/" + k)
                        episode["action/" + k].append(v)

                # All steps in this episode failed to decode.
                if idx < 0 or not videos:
                    print(
                        f"Skipping episode {episode_id}: no valid decoded steps remained after filtering decode errors."
                    )
                    episode_id += 1
                    pbar.update(1)
                    continue

                for k in list(segments.keys()):
                    if not segments[k]:
                        del segments[k]
                        continue
                    segments[k].append((idx + 1, ""))
                if not segments:
                    segments["not_captioned"] = [(0, ""), (idx + 1, "")]

                for view, frames in videos.items():
                    frames = np.stack(frames)
                    n, h, w, _ = frames.shape
                    video_path = episode_dir / f"{view}.mp4"

                    if h % 2 != 0:
                        h = h - 1
                        frames = frames[:, :h, :, :]
                    if w % 2 != 0:
                        w = w - 1
                        frames = frames[:, :, :w, :]
                    write_numpy_to_mp4(frames, str(video_path))

                    for k, v in segments.items():
                        for s in range(len(v) - 1):
                            start_idx, caption = v[s]
                            end_idx = v[s + 1][0]
                            record = dict(
                                video_path=str(video_path.relative_to(self.data_root)),
                                state_path=str(state_path.relative_to(self.data_root)),
                                height=h,
                                width=w,
                                n_frames=end_idx - start_idx,
                                trim_start=start_idx,
                                trim_end=end_idx,
                                fps=self.cfg.download.openx_fps,
                                original_caption=caption,
                                has_caption=v[0][1] != "",
                            )
                            episode_records[view].append(record)
                for view, records in episode_records.items():
                    df = pd.DataFrame.from_records(records)
                    df.to_csv(episode_dir / f"{view}.csv", index=False)

                for k in fields_to_stack:
                    episode[k] = np.stack(episode[k])
                with open(state_path, "wb") as f:
                    pickle.dump(episode, f)
            except decode_errors as e:
                print(
                    f"Skipping episode {episode_id} due to decode error while processing steps: {e}"
                )
                episode_id += 1
                pbar.update(1)
                continue
            episode_id += 1
            pbar.update(1)
        pbar.close()

        # Save metadata
        metadata_path = self.data_root / self.metadata_path
        metadata_dir = metadata_path.parent
        metadata_dir.mkdir(parents=True, exist_ok=True)
        record_dict = defaultdict(list)
        for episode_dir in all_episode_dir.glob("episode_*"):
            for view_csv in episode_dir.glob("*.csv"):
                view_csv = view_csv.name
                view_df = pd.read_csv(episode_dir / view_csv)
                record_dict[view_csv].extend(view_df.to_dict("records"))
        all_df = []
        for view_csv, records in record_dict.items():
            df = pd.DataFrame.from_records(records)
            df.to_csv(metadata_dir / view_csv, index=False)
            print(
                f"Created metadata csv for view {view_csv.split('.')[0]} with {len(df)} records"
            )
            if view_csv.replace(".csv", "") in self.cfg.download.views:
                all_df.append(df)
        all_df = pd.concat(all_df)
        all_df.to_csv(metadata_path, index=False)
        print(f"Created metadata CSV with {len(all_df)} records")
