# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training wrapper for environments without torchvision.io.VideoReader.

Why this wrapper is needed:
    LeRobot's video_backend="pyav" still routes through torchvision.io.VideoReader
    internally (video_utils.py L151-152). On platforms where VideoReader is not
    available (e.g. aarch64, certain GPU-less builds), lerobot_train crashes with:
        AttributeError: module 'torchvision.io' has no attribute 'VideoReader'

    This script monkey-patches decode_video_frames to use PyAV directly,
    bypassing VideoReader entirely.

Usage:
    See Makefile for standard training commands, or run directly:

    python train_realman.py \\
        --dataset.repo_id=robocoin/Realman_RMC_AIDA_L_storage_block_basket \\
        --dataset.root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \\
        --policy.type=act \\
        --policy.push_to_hub=false \\
        --num_workers=0 \\
        --batch_size=8 \\
        --steps=100000 \\
        --output_dir=outputs/act_realman
"""

import sys
import os

# Add lerobot to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lerobot", "src"))

import av
import torch

# Monkey-patch video decoding before importing lerobot
import lerobot.datasets.video_utils as video_utils


def decode_video_frames_av(
    video_path,
    timestamps,
    tolerance_s,
    backend=None,
):
    """Decode video frames using av library directly.

    This is a fallback implementation for platforms where torchvision.io.VideoReader
    and torchcodec are not available. It uses PyAV to decode frames and matches
    them to the requested timestamps.

    Args:
        video_path: Path to the video file.
        timestamps: List of timestamps (in seconds) to extract frames at.
        tolerance_s: Maximum allowed deviation between requested and actual timestamps.
        backend: Ignored (kept for API compatibility).

    Returns:
        torch.Tensor of shape (N, C, H, W) with float32 values in [0, 1].
    """
    video_path = str(video_path)

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)

        first_ts = min(timestamps)
        last_ts = max(timestamps)

        # Seek to a keyframe before the first requested timestamp.
        # Use a generous buffer (1s) to ensure we land before the target frame,
        # since seek goes to the nearest keyframe before the target.
        seek_ts = max(0, first_ts - 1.0)
        container.seek(int(seek_ts / time_base), stream=stream)

        loaded_frames = []
        loaded_ts = []
        for frame in container.decode(video=0):
            pts = frame.pts * time_base
            if pts > last_ts + tolerance_s:
                break
            img = frame.to_ndarray(format="rgb24")
            tensor = torch.from_numpy(img).permute(2, 0, 1)
            loaded_frames.append(tensor)
            loaded_ts.append(pts)

    if not loaded_frames:
        raise video_utils.FrameTimestampError(
            f"No frames loaded from {video_path} for timestamps {timestamps}"
        )

    query_ts = torch.tensor(timestamps, dtype=torch.float64)
    loaded_ts_t = torch.tensor(loaded_ts, dtype=torch.float64)

    # Find closest loaded frame for each query timestamp
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_dist, argmin_ = dist.min(1)

    is_within_tol = min_dist < tolerance_s
    if not is_within_tol.all():
        raise video_utils.FrameTimestampError(
            f"Timestamps violate tolerance ({min_dist[~is_within_tol]} > {tolerance_s=}).\n"
            f"queried: {query_ts}\nloaded: {loaded_ts_t}\nvideo: {video_path}"
        )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_frames = closest_frames.float() / 255.0

    return closest_frames


# Apply monkey-patch
video_utils.decode_video_frames = decode_video_frames_av

# Now import and run training
from lerobot.scripts.lerobot_train import train

if __name__ == "__main__":
    train()
