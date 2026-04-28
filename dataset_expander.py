"""
dataset_expander.py — Automatically download AI-generated images for training.

Downloads from Kaggle datasets and organises into:
    dataset/fake/hard_fake/      (AI-generated face portraits)
    dataset/fake/hard_fake_scene/ (AI-generated scene images with people)

Prerequisites:
    pip install kaggle Pillow

    You must have a valid Kaggle API token at ~/.kaggle/kaggle.json
    (or %USERPROFILE%\\.kaggle\\kaggle.json on Windows).
    Get one from: https://www.kaggle.com/settings → API → Create New Token

Usage:
    python dataset_expander.py
    python dataset_expander.py --dataset dataset --min-images 1500
    python dataset_expander.py --skip-download        # just verify existing
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Kaggle datasets to download.
#
# Each entry:   (kaggle_dataset_slug, category)
#   category = "face" → goes to hard_fake/
#   category = "scene" → goes to hard_fake_scene/
#
# These are well-known public datasets of AI-generated images.
# ---------------------------------------------------------------------------
KAGGLE_SOURCES = [
    # AI-generated faces (diffusion / GAN)
    ("xhlulu/flickrfaceshq-dataset-nvidia-resized-256px", "face"),
    ("ciplab/real-and-fake-face-detection", "face"),
    ("aldiputr/140k-real-and-fake-faces", "face"),
    # AI-generated scene / lifestyle images
    ("superpotato9/dalle-recognition-dataset", "scene"),
    ("succinctlyai/midjourney-texttoimage", "scene"),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
MIN_IMAGE_SIZE = (64, 64)  # reject tiny thumbnails


def _ensure_kaggle():
    """Check kaggle CLI is available, install if missing. Set up credentials."""
    try:
        subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[EXPANDER] Installing kaggle CLI...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

    # Support KAGGLE_API_TOKEN env var (new token format: KGAT_...)
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        print(f"[EXPANDER] Using KAGGLE_API_TOKEN env var")
        return

    # Check for kaggle.json
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if kaggle_json.exists():
        print(f"[EXPANDER] Using credentials from {kaggle_json}")
        return

    print(
        "[EXPANDER] No Kaggle credentials found.\n"
        "  Option 1: Set KAGGLE_API_TOKEN environment variable\n"
        "  Option 2: Place kaggle.json in ~/.kaggle/\n"
        "  Get token from: https://www.kaggle.com/settings → API"
    )


def _download_kaggle_dataset(slug: str, dest: Path) -> bool:
    """Download and unzip a Kaggle dataset. Returns True if successful."""
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / f".downloaded_{slug.replace('/', '_')}"
    if marker.exists():
        print(f"[EXPANDER] Already downloaded: {slug}")
        return True

    print(f"[EXPANDER] Downloading: {slug} → {dest}")
    try:
        subprocess.check_call(
            [
                "kaggle", "datasets", "download",
                "-d", slug,
                "-p", str(dest),
                "--unzip",
            ],
            timeout=600,
        )
        marker.touch()
        return True
    except subprocess.CalledProcessError as e:
        print(f"[EXPANDER] FAILED to download {slug}: {e}")
        return False
    except subprocess.TimeoutExpired:
        print(f"[EXPANDER] TIMEOUT downloading {slug}")
        return False


def _file_hash(path: Path) -> str:
    """Quick 8-byte hash for dedup."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(8192))
    return h.hexdigest()[:16]


def _is_valid_image(path: Path) -> bool:
    """Check if file is a valid image above minimum size."""
    try:
        with Image.open(path) as img:
            w, h = img.size
            return w >= MIN_IMAGE_SIZE[0] and h >= MIN_IMAGE_SIZE[1]
    except Exception:
        return False


def _collect_images(root: Path):
    """Recursively collect image paths."""
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            yield f


def _copy_images_dedup(
    source_dir: Path,
    target_dir: Path,
    seen_hashes: set,
    max_images: int = 2000,
    label: str = "",
) -> int:
    """Copy images from source_dir into target_dir, deduplicating by hash."""
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img_path in _collect_images(source_dir):
        if copied >= max_images:
            break
        if not _is_valid_image(img_path):
            continue
        h = _file_hash(img_path)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        ext = img_path.suffix.lower()
        dst = target_dir / f"{label}_{copied:05d}{ext}"
        shutil.copy2(img_path, dst)
        copied += 1

    return copied


def expand_dataset(
    dataset_root: str = "dataset",
    min_images: int = 1000,
    skip_download: bool = False,
):
    """Main: download AI images and organize into dataset structure."""
    root = Path(dataset_root)
    hard_fake_dir = root / "fake" / "hard_fake"
    hard_fake_scene_dir = root / "fake" / "hard_fake_scene"
    hard_fake_dir.mkdir(parents=True, exist_ok=True)
    hard_fake_scene_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images
    existing_hf = len(list(_collect_images(hard_fake_dir)))
    existing_hfs = len(list(_collect_images(hard_fake_scene_dir)))
    existing = existing_hf + existing_hfs
    print(f"[EXPANDER] Existing: hard_fake={existing_hf}, hard_fake_scene={existing_hfs}")

    if existing >= min_images and skip_download:
        print(f"[EXPANDER] Already have {existing} images (≥ {min_images}). Done.")
        return True

    if skip_download:
        print(f"[EXPANDER] Skip download. Have {existing}/{min_images} images.")
        return existing >= min_images

    # Ensure kaggle CLI
    _ensure_kaggle()

    # Build dedup set from existing images
    seen_hashes = set()
    for img in _collect_images(hard_fake_dir):
        seen_hashes.add(_file_hash(img))
    for img in _collect_images(hard_fake_scene_dir):
        seen_hashes.add(_file_hash(img))

    staging = root / "_staging_downloads"
    staging.mkdir(parents=True, exist_ok=True)

    total_new = 0
    need = max(0, min_images - existing)

    for slug, category in KAGGLE_SOURCES:
        if total_new >= need:
            break

        dl_dir = staging / slug.replace("/", "_")
        success = _download_kaggle_dataset(slug, dl_dir)
        if not success:
            continue

        # Route to correct target
        if category == "face":
            target = hard_fake_dir
            label = "hf"
        else:
            target = hard_fake_scene_dir
            label = "hfs"

        per_source_max = max(500, (need - total_new))
        n = _copy_images_dedup(dl_dir, target, seen_hashes, per_source_max, label)
        total_new += n
        print(f"[EXPANDER] Copied {n} images from {slug} → {target.name}")

    # Cleanup staging
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)

    # Final count
    final_hf = len(list(_collect_images(hard_fake_dir)))
    final_hfs = len(list(_collect_images(hard_fake_scene_dir)))
    total = final_hf + final_hfs
    print(f"\n[EXPANDER] Final: hard_fake={final_hf}, hard_fake_scene={final_hfs}, total={total}")

    if total < min_images:
        print(
            f"[WARN] Only {total} images collected ({min_images} target).\n"
            "       You may need to manually add more images to:\n"
            f"         {hard_fake_dir}\n"
            f"         {hard_fake_scene_dir}\n"
            "       Or provide Kaggle credentials (kaggle.json)."
        )
        return False

    print(f"[EXPANDER] ✓ Dataset expanded successfully with {total} hard_fake images.")
    return True


def verify_no_fakes_in_real(dataset_root: str = "dataset"):
    """Safety check: ensure real/ contains no paths from fake sources."""
    real_dir = Path(dataset_root) / "real"
    if not real_dir.exists():
        return True

    suspicious = []
    fake_keywords = {"fake", "synthetic", "ai_generated", "gan", "deepfake", "diffusion"}
    for img in _collect_images(real_dir):
        parts_lower = [p.lower() for p in img.parts]
        for kw in fake_keywords:
            if kw in parts_lower:
                suspicious.append(str(img))
                break

    if suspicious:
        print(f"[WARN] Found {len(suspicious)} suspicious files in real/:")
        for s in suspicious[:10]:
            print(f"  - {s}")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand dataset with AI-generated images.")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset root directory.")
    parser.add_argument("--min-images", type=int, default=1000, help="Minimum total hard_fake images.")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, just verify.")
    args = parser.parse_args()

    ok = expand_dataset(
        dataset_root=args.dataset,
        min_images=args.min_images,
        skip_download=args.skip_download,
    )
    verify_no_fakes_in_real(args.dataset)

    sys.exit(0 if ok else 1)
