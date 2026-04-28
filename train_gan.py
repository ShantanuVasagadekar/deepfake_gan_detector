"""
train_gan.py — Full GAN adversarial training pipeline.

Trains Generator and Discriminator simultaneously on the deepfake dataset.
After training the **discriminator** is saved as the deepfake detection model.

Features:
  • GPU acceleration (CUDA) + mixed‐precision (AMP)
  • Multi‐threaded data loading
  • Model checkpointing every N epochs
  • Resume‐from‐checkpoint
  • Early stopping on D‐accuracy plateau
  • Loss tracking + training curve plots

Usage:
    python train_gan.py --epochs 50 --batch-size 64 --dataset dataset
    python train_gan.py --resume saved_models/checkpoint_epoch10.pth
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from gan_model import Generator, Discriminator, NZ
from face_preprocessing import get_dataloader
from utils import (
    get_device,
    weights_init,
    save_checkpoint,
    load_checkpoint,
    ensure_dir,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SAVE_DIR = Path(__file__).resolve().parent / "saved_models"
LOG_FILE = Path(__file__).resolve().parent / "training.log"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.0002,
    beta1: float = 0.5,
    dataset_root: str = "dataset",
    image_size: int = 128,
    num_workers: int = 4,
    checkpoint_every: int = 5,
    resume_path: str = None,
    patience: int = 10,
    max_per_class: int = None,
):
    """Run the full adversarial training pipeline."""
    logger = setup_logging(str(LOG_FILE))
    device = get_device()
    ensure_dir(str(SAVE_DIR))

    # ── Data ───────────────────────────────────────────────────────────────
    loader = get_dataloader(
        root=dataset_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train=True,
        max_per_class=max_per_class,
    )
    logger.info(f"DataLoader ready  |  batches/epoch: {len(loader)}")

    # ── Models ─────────────────────────────────────────────────────────────
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    scaler = GradScaler()  # mixed precision

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch = 0
    history = {"G_loss": [], "D_loss": [], "D_acc": []}

    if resume_path and os.path.isfile(resume_path):
        ckpt = load_checkpoint(resume_path, device)
        netG.load_state_dict(ckpt["netG"])
        netD.load_state_dict(ckpt["netD"])
        optimizerG.load_state_dict(ckpt["optimizerG"])
        optimizerD.load_state_dict(ckpt["optimizerD"])
        start_epoch = ckpt.get("epoch", 0) + 1
        history = ckpt.get("history", history)
        logger.info(f"Resumed from epoch {start_epoch}")

    fixed_noise = torch.randn(16, NZ, 1, 1, device=device)

    # ── Early stopping state ───────────────────────────────────────────────
    best_d_acc = 0.0
    stall_count = 0

    # ── Labels ─────────────────────────────────────────────────────────────
    real_label = 1.0
    fake_label = 0.0

    logger.info("=" * 60)
    logger.info(f"Starting training  |  epochs {start_epoch}→{epochs}")
    logger.info("=" * 60)
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_d_correct = 0
        epoch_total = 0

        for i, (real_imgs, _) in enumerate(loader):
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # Labels
            label_real = torch.full((b_size,), real_label, device=device)
            label_fake = torch.full((b_size,), fake_label, device=device)

            # ────────── Update Discriminator ──────────
            netD.zero_grad()
            with autocast():
                # Real batch
                output_real = netD(real_imgs).view(-1)
                lossD_real = criterion(output_real, label_real)

                # Fake batch
                noise = torch.randn(b_size, NZ, 1, 1, device=device)
                fake_imgs = netG(noise)
                output_fake = netD(fake_imgs.detach()).view(-1)
                lossD_fake = criterion(output_fake, label_fake)

                lossD = lossD_real + lossD_fake

            scaler.scale(lossD).backward()
            scaler.step(optimizerD)

            # Accuracy bookkeeping
            preds_real = (output_real > 0.5).float()
            preds_fake = (output_fake <= 0.5).float()
            epoch_d_correct += preds_real.sum().item() + preds_fake.sum().item()
            epoch_total += 2 * b_size

            # ────────── Update Generator ──────────
            netG.zero_grad()
            with autocast():
                output = netD(fake_imgs).view(-1)
                lossG = criterion(output, label_real)  # fool D

            scaler.scale(lossG).backward()
            scaler.step(optimizerG)

            scaler.update()

            epoch_d_loss += lossD.item()
            epoch_g_loss += lossG.item()

        # ── Epoch stats ───────────────────────────────────────────────────
        n_batches = max(len(loader), 1)
        avg_d_loss = epoch_d_loss / n_batches
        avg_g_loss = epoch_g_loss / n_batches
        d_acc = epoch_d_correct / max(epoch_total, 1)

        history["D_loss"].append(avg_d_loss)
        history["G_loss"].append(avg_g_loss)
        history["D_acc"].append(d_acc)

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"D_loss: {avg_d_loss:.4f}  G_loss: {avg_g_loss:.4f}  "
            f"D_acc: {d_acc * 100:.1f}%"
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
            ckpt_path = str(SAVE_DIR / f"checkpoint_epoch{epoch + 1}.pth")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "netG": netG.state_dict(),
                    "netD": netD.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                    "optimizerD": optimizerD.state_dict(),
                    "history": history,
                },
                ckpt_path,
            )

        # ── Early stopping ────────────────────────────────────────────────
        if d_acc > best_d_acc:
            best_d_acc = d_acc
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(D_acc stalled for {patience} epochs)"
                )
                break

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed / 60:.1f} min")

    # ── Save final discriminator as detector ──────────────────────────────
    detector_path = str(SAVE_DIR / "deepfake_detector.pth")
    torch.save(netD.state_dict(), detector_path)
    logger.info(f"Deepfake detector saved → {detector_path}")

    # ── Plot losses ───────────────────────────────────────────────────────
    _plot_training_curves(history)

    return history


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def _plot_training_curves(history: dict):
    """Save loss and accuracy curves to disk."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["G_loss"], label="Generator Loss", color="#e74c3c")
    ax1.plot(history["D_loss"], label="Discriminator Loss", color="#3498db")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("GAN Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        [a * 100 for a in history["D_acc"]],
        label="Discriminator Accuracy",
        color="#2ecc71",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Discriminator Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = str(SAVE_DIR.parent / "training_loss.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Training curves → {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DCGAN for deepfake detection.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs).")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap images per class (for quick testing).")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset_root=args.dataset,
        image_size=args.image_size,
        num_workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        resume_path=args.resume,
        patience=args.patience,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
