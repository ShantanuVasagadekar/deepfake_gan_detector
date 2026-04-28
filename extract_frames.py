"""
extract_frames.py — Extract frames from video files in the dataset.

Walks the dataset directory for video files (.mp4, .avi, .mov, .mkv),
extracts every N-th frame, and saves them as JPEG images alongside the
source videos.

Usage:
    python extract_frames.py --input dataset/fake --interval 10
"""

import os
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frame_interval: int = 10,
) -> int:
    """
    Extract frames from a single video file.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        frame_interval: Save one frame every *frame_interval* frames.

    Returns:
        Number of frames saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return 0

    video_name = Path(video_path).stem
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            fname = f"{video_name}_frame{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, fname), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def extract_all(input_dir: str, frame_interval: int = 10):
    """
    Recursively find all videos under *input_dir* and extract frames.

    Extracted frames are written to a ``frames/`` sub-directory next to
    each video file.
    """
    input_path = Path(input_dir)
    videos = [
        p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not videos:
        print(f"[INFO] No video files found under {input_dir}")
        return

    print(f"[INFO] Found {len(videos)} video(s) under {input_dir}")
    total_frames = 0

    for vp in tqdm(videos, desc="Extracting frames", unit="video"):
        out_dir = vp.parent / "frames"
        n = extract_frames_from_video(str(vp), str(out_dir), frame_interval)
        total_frames += n

    print(f"[DONE] Extracted {total_frames:,} frames total.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video files in the dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset",
        help="Root directory containing video files (default: dataset).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Extract 1 frame every N frames (default: 10).",
    )
    args = parser.parse_args()
    extract_all(args.input, args.interval)


if __name__ == "__main__":
    main()
