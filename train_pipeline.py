"""
train_pipeline.py — End-to-end training automation.

Steps:
    1. Expand dataset (download AI images if needed)
    2. Verify dataset structure
    3. Auto-detect GPU memory → adjust batch size
    4. Launch training

Usage:
    python train_pipeline.py
    python train_pipeline.py --dataset dataset --epochs 60 --skip-download
    python train_pipeline.py --batch-size 16     # force smaller batch
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def count_images(directory: Path) -> int:
    """Count image files recursively."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)


def verify_dataset(dataset_root: str) -> dict:
    """Verify dataset structure and print statistics."""
    root = Path(dataset_root)

    dirs = {
        "real": root / "real",
        "fake": root / "fake",
        "hard_fake": root / "fake" / "hard_fake",
        "hard_fake_scene": root / "fake" / "hard_fake_scene",
    }

    counts = {}
    for name, d in dirs.items():
        counts[name] = count_images(d)

    # Count other fakes (everything in fake/ except hard_fake dirs)
    total_fake = counts["fake"]
    hard_total = counts["hard_fake"] + counts["hard_fake_scene"]
    other_fake = total_fake - hard_total

    print("\n" + "=" * 50)
    print("  DATASET STRUCTURE VERIFICATION")
    print("=" * 50)
    print(f"  Real images      : {counts['real']:,}")
    print(f"  Total fake       : {total_fake:,}")
    print(f"    ├─ hard_fake   : {counts['hard_fake']:,}")
    print(f"    ├─ hard_fake_scene: {counts['hard_fake_scene']:,}")
    print(f"    └─ other fake  : {other_fake:,}")
    print(f"  Total            : {counts['real'] + total_fake:,}")

    if total_fake > 0:
        hf_pct = hard_total / total_fake * 100
        print(f"  hard_fake %      : {hf_pct:.1f}% of fake data", end="")
        if hf_pct >= 20:
            print(" ✓")
        else:
            print(f" ✗ (target ≥ 20%)")
    print("=" * 50 + "\n")

    issues = []
    if counts["real"] == 0:
        issues.append("No real images found in dataset/real/")
    if total_fake == 0:
        issues.append("No fake images found in dataset/fake/")
    if hard_total == 0:
        issues.append("No hard_fake images found. Run dataset_expander.py first.")
    if total_fake > 0 and hard_total / total_fake < 0.20:
        issues.append(
            f"hard_fake is only {hard_total}/{total_fake} = "
            f"{hard_total/total_fake*100:.1f}% of fake (target ≥ 20%)"
        )

    for issue in issues:
        print(f"  [WARN] {issue}")

    return {"counts": counts, "issues": issues}


def detect_gpu_batch_size(requested: int) -> int:
    """Auto-reduce batch size if GPU memory is limited."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("[PIPELINE] No GPU detected → using CPU (batch_size unchanged)")
            return requested

        gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[PIPELINE] GPU: {gpu_name}, Memory: {gpu_mem_gb:.1f} GB")

        # 4-branch model needs ~3GB for batch_size=32, ~1.8GB for batch_size=16
        if gpu_mem_gb < 4.0 and requested > 16:
            print(f"[PIPELINE] Low GPU memory ({gpu_mem_gb:.1f}GB) → reducing batch_size {requested} → 16")
            return 16
        elif gpu_mem_gb < 6.0 and requested > 24:
            print(f"[PIPELINE] Medium GPU memory ({gpu_mem_gb:.1f}GB) → reducing batch_size {requested} → 24")
            return 24

        return requested
    except ImportError:
        return requested


def run_pipeline(
    dataset_root: str = "dataset",
    epochs: int = 60,
    batch_size: int = 32,
    contrastive_weight: float = 0.3,
    hard_fake_mult: int = 4,
    other_fake_mult: int = 2,
    fake_bucket_cap: int = 4000,
    skip_download: bool = False,
    skip_verify: bool = False,
    min_images: int = 1000,
    val_unseen: str = None,
):
    """Run the full training pipeline."""

    print("\n" + "=" * 60)
    print("  DEEPSHIELD V3 TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Dataset expansion
    print("\n[STEP 1/3] Dataset Expansion")
    print("-" * 40)

    expander_script = Path(__file__).resolve().parent / "dataset_expander.py"
    if not expander_script.exists():
        print(f"[ERROR] dataset_expander.py not found at {expander_script}")
        return False

    expander_cmd = [
        sys.executable, str(expander_script),
        "--dataset", dataset_root,
        "--min-images", str(min_images),
    ]
    if skip_download:
        expander_cmd.append("--skip-download")

    result = subprocess.run(expander_cmd)
    if result.returncode != 0 and not skip_download:
        print("[WARN] Dataset expansion had issues. Continuing with available data...")

    # Step 2: Verify dataset
    print("\n[STEP 2/3] Dataset Verification")
    print("-" * 40)

    if not skip_verify:
        info = verify_dataset(dataset_root)
        total = info["counts"]["real"] + info["counts"]["fake"]
        if total < 10:
            print("[ERROR] Dataset too small. Add images and retry.")
            return False
        if info["issues"]:
            print("[PIPELINE] Continuing despite warnings...")
    else:
        print("  (skipped)")

    # Step 3: Training
    print("\n[STEP 3/3] Training")
    print("-" * 40)

    # Auto-adjust batch size for GPU
    batch_size = detect_gpu_batch_size(batch_size)

    train_script = Path(__file__).resolve().parent / "train_classifier.py"
    if not train_script.exists():
        print(f"[ERROR] train_classifier.py not found at {train_script}")
        return False

    train_cmd = [
        sys.executable, str(train_script),
        "--dataset", dataset_root,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--contrastive-weight", str(contrastive_weight),
        "--hard-fake-mult", str(hard_fake_mult),
        "--other-fake-mult", str(other_fake_mult),
        "--fake-bucket-cap", str(fake_bucket_cap),
        "--balance-fake-sources",
    ]
    if val_unseen:
        train_cmd.extend(["--val-unseen", val_unseen])

    print(f"\n[PIPELINE] Launching training:")
    print(f"  {' '.join(train_cmd[2:])}\n")

    result = subprocess.run(train_cmd)
    if result.returncode != 0:
        print("[ERROR] Training failed.")
        return False

    # Check output
    v3_model = Path(dataset_root).parent / "saved_models" / "deepfake_classifier_v3.pth"
    # Also check relative to script dir
    script_dir = Path(__file__).resolve().parent
    v3_alt = script_dir / "saved_models" / "deepfake_classifier_v3.pth"

    if v3_model.exists() or v3_alt.exists():
        model_path = v3_model if v3_model.exists() else v3_alt
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\n[PIPELINE] ✓ Model saved: {model_path} ({size_mb:.1f} MB)")
    else:
        print("[PIPELINE] ✓ Training complete. Check saved_models/ for output.")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="End-to-end deepfake training pipeline.")
    p.add_argument("--dataset", type=str, default="dataset", help="Dataset root directory.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--contrastive-weight", type=float, default=0.3)
    p.add_argument("--hard-fake-mult", type=int, default=4)
    p.add_argument("--other-fake-mult", type=int, default=2)
    p.add_argument("--fake-bucket-cap", type=int, default=4000)
    p.add_argument("--skip-download", action="store_true", help="Skip dataset download step.")
    p.add_argument("--skip-verify", action="store_true", help="Skip dataset verification.")
    p.add_argument("--min-images", type=int, default=1000, help="Min hard_fake images to collect.")
    p.add_argument("--val-unseen", type=str, default=None, help="Path to unseen validation set.")
    args = p.parse_args()

    success = run_pipeline(
        dataset_root=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        contrastive_weight=args.contrastive_weight,
        hard_fake_mult=args.hard_fake_mult,
        other_fake_mult=args.other_fake_mult,
        fake_bucket_cap=args.fake_bucket_cap,
        skip_download=args.skip_download,
        skip_verify=args.skip_verify,
        min_images=args.min_images,
        val_unseen=args.val_unseen,
    )

    sys.exit(0 if success else 1)
