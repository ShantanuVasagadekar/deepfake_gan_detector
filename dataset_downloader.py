"""
dataset_downloader.py — Automated dataset download via Kaggle API.

Downloads and organizes:
  Real  → FFHQ, CelebA-HQ, Human Faces (Real)
  Fake  → Celeb-DF v2, DFDC, StyleGAN, AI-Generated Faces,
          ThisPersonDoesNotExist, DiffusionFaces

Final layout:
    dataset/
      real/
        ffhq/
        celebahq/
        human_faces/
      fake/
        celebdf/
        dfdc/
        stylegan/
        ai_generated/
        diffusion/
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KAGGLE_USERNAME = "shantanuvasagadekar"
KAGGLE_KEY = "8a5b0a3311fab9eda5123298ce2e8f20"

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

# Kaggle dataset slugs → (category, subfolder)
# These datasets provide a mix of real and synthetic face images.
DATASETS = {
    # ── Real faces ──────────────────────────────────────────────
    "kaustubhdhote/human-faces-dataset": ("staging", "human_faces"),

    # ── Fake / AI-generated faces ──────────────────────────────
    "ciplab/real-and-fake-face-detection": ("fake", "ciplab_faces"),
    "xhlulu/140k-real-and-fake-faces": ("staging", "140k_faces"),
    "hamzameer/fake-vs-real-faces-tpdne": ("staging", "tpdne"),
    "manjilkarki/deepfake-and-real-images": ("staging", "deepfake_real"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure_kaggle():
    """Write Kaggle credentials so the kaggle library can authenticate."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    cred_file = kaggle_dir / "kaggle.json"
    creds = {"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}
    cred_file.write_text(json.dumps(creds))
    try:
        os.chmod(cred_file, 0o600)
    except Exception:
        pass
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    print("[INFO] Kaggle credentials configured.")


def _download_dataset(slug: str, dest: Path):
    """Download and extract a single Kaggle dataset into dest."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n[DOWNLOAD] {slug} → {dest}")
    try:
        api.dataset_download_files(slug, path=str(dest), unzip=True)
        print(f"[DONE]     {slug}")
    except Exception as e:
        print(f"[ERROR]    Could not download {slug}: {e}")


def _extract_existing_zips(root: Path):
    """Find and extract any .zip files that were downloaded but not extracted."""
    print("\n[EXTRACT] Checking for unextracted ZIP files...")
    for zip_path in root.rglob("*.zip"):
        print(f"  Extracting {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(zip_path.parent)
            print(f"  [DONE] Extracted {zip_path.name}")
        except Exception as e:
            print(f"  [ERROR] Could not extract {zip_path.name}: {e}")


def _organize_staging(staging_dir: Path, real_dir: Path, fake_dir: Path):
    """
    Organize staging datasets into real/ and fake/ folders.
    Searches for common subfolder names like 'real', 'fake', 'training_real', etc.
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    real_keywords = ["real", "genuine", "authentic", "training_real", "Real Images"]
    fake_keywords = ["fake", "synthetic", "generated", "ai", "training_fake",
                     "AI-Generated", "AI-Generated Images", "deepfake"]

    for item in staging_dir.iterdir():
        if not item.is_dir():
            continue

        # Walk the subdirectory tree looking for real/fake folders
        for subdir in item.rglob("*"):
            if not subdir.is_dir():
                continue

            name_lower = subdir.name.lower()

            target = None
            if any(kw.lower() in name_lower for kw in real_keywords):
                target = real_dir / item.name
            elif any(kw.lower() in name_lower for kw in fake_keywords):
                target = fake_dir / item.name

            if target is not None:
                target.mkdir(parents=True, exist_ok=True)
                images = [f for f in subdir.iterdir()
                          if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
                for img_file in images:
                    dest = target / img_file.name
                    if not dest.exists():
                        shutil.copy2(str(img_file), str(dest))

    print(f"[ORGANIZE] Moved staging images to real/ and fake/")


def _count_images(root: Path) -> int:
    """Count image files recursively under root."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    count = 0
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def download_all():
    """Download every dataset listed in DATASETS."""
    _configure_kaggle()

    for slug, (category, folder) in DATASETS.items():
        dest = DATASET_DIR / category / folder
        if dest.exists() and _count_images(dest) > 100:
            print(f"[SKIP] {slug} already downloaded ({_count_images(dest):,} images)")
            continue
        _download_dataset(slug, dest)

    _extract_existing_zips(DATASET_DIR)

    # Organize staging datasets into real/fake
    staging = DATASET_DIR / "staging"
    if staging.exists():
        _organize_staging(
            staging,
            DATASET_DIR / "real",
            DATASET_DIR / "fake"
        )

    # Summary
    real_count = _count_images(DATASET_DIR / "real")
    fake_count = _count_images(DATASET_DIR / "fake")
    total = real_count + fake_count
    print("\n" + "=" * 55)
    print(f"  Real images  : {real_count:>8,}")
    print(f"  Fake images  : {fake_count:>8,}")
    print(f"  Total        : {total:>8,}")
    print("=" * 55)


if __name__ == "__main__":
    download_all()
