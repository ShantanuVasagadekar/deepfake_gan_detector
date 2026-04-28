import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classifier_model import DeepfakeClassifier
from face_preprocessing import (
    discover_dataset_samples,
    balance_class_samples,
    stratified_indices,
    SampleListDataset,
    get_eval_transforms,
    LABEL_FAKE,
)
from utils import get_device, ensure_dir, setup_logging

SAVE_DIR = Path(__file__).resolve().parent / "saved_models"
LOG_FILE = Path(__file__).resolve().parent / "classifier_training.log"
IMG_SIZE = 256


class IndexedWrapper(Dataset):
    def __init__(self, base: SampleListDataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx


class FocalLossDeepfake(nn.Module):
    def __init__(
        self,
        gamma: float = 2.5,
        alpha_fake: float = 0.78,
        label_smoothing: float = 0.04,
        fn_soft_scale: float = 2.2,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha_fake = alpha_fake
        self.alpha_real = 1.0 - alpha_fake
        self.label_smoothing = label_smoothing
        self.fn_soft_scale = fn_soft_scale

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        t = targets.float()
        if self.label_smoothing > 0:
            t = t * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * t + (1 - probs) * (1 - t)
        p_t = p_t.clamp(1e-6, 1 - 1e-6)
        focal = (1 - p_t) ** self.gamma
        alpha_t = self.alpha_fake * t + self.alpha_real * (1 - t)
        loss = alpha_t * focal * bce
        fn_aware = 1.0 + self.fn_soft_scale * t * (1 - probs)
        loss = loss * fn_aware
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    alpha: float,
    device: torch.device,
):
    if alpha <= 0:
        return x, y, w, None
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=device)
    x2 = x[idx]
    y2 = y[idx]
    w2 = w[idx]
    x_mix = lam * x + (1 - lam) * x2
    y_mix = lam * y + (1 - lam) * y2
    w_mix = torch.maximum(w, w2)
    return x_mix, y_mix, w_mix, lam


def _fake_recall_metrics(labels_list, preds_list, probs_list):
    labels_t = torch.tensor(labels_list).float()
    preds_t = torch.tensor(preds_list).float()
    fake_mask = labels_t > 0.5
    n_fake = fake_mask.sum().clamp(min=1.0)
    fake_rec = ((preds_t > 0.5) & fake_mask).sum() / n_fake
    n_real = (~fake_mask).sum().clamp(min=1.0)
    real_ok = ((preds_t < 0.5) & ~fake_mask).sum() / n_real
    acc = (preds_t.eq(labels_t)).sum() / len(labels_t)
    return fake_rec.item(), real_ok.item(), acc.item()


def _eval_loader_metrics(model, loader, device, criterion):
    model.eval()
    tot_loss = 0.0
    n = 0
    labels_all, preds_all, probs_all = [], [], []
    use_amp = (device.type == "cuda")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1).float()
            with autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            tot_loss += loss.item() * images.size(0)
            n += images.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            labels_all.extend(labels.view(-1).cpu().tolist())
            preds_all.extend(preds.view(-1).cpu().tolist())
            probs_all.extend(probs.view(-1).cpu().tolist())
    fr, rr, acc = _fake_recall_metrics(labels_all, preds_all, probs_all)
    return tot_loss / max(n, 1), fr, rr, acc


def _plot_history(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train", linewidth=2)
    ax1.plot(history["val_loss"], label="Val", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(history["val_fake_recall"], label="Val fake recall", linewidth=2)
    ax2.plot(history["val_acc"], label="Val acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plot_path = SAVE_DIR.parent / "classifier_curves.png"
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Curves saved to {plot_path}", flush=True)


def train(
    dataset_root: str = r"C:\Users\Shantanu Vasagadekar\Desktop\deepfake detection (PBEL)\deepfake_gan_detector\dataset",
    epochs: int = 15,
    batch_size: int = 48,
    lr: float = 1.8e-4,
    weight_decay: float = 1.2e-4,
    max_per_class: int = None,
    val_ratio: float = 0.2,
    mixup_alpha: float = 0.0,
    fake_sampler_boost: float = 1.2,
    hard_mine_mult: float = 1.3,
    hard_mine_cap: float = 4.0,
):
    logger = setup_logging(str(LOG_FILE))
    device = get_device()
    ensure_dir(str(SAVE_DIR))

    # GPU performance optimizations
    use_amp = (device.type == "cuda")
    if use_amp:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print(f"\n{'='*60}", flush=True)
    print(f"  Deepfake Classifier Training (GPU Optimized)", flush=True)
    print(f"  Epochs: {epochs}  |  Batch size: {batch_size}  |  Device: {device}", flush=True)
    print(f"  AMP: {use_amp}  |  cuDNN benchmark: {use_amp}", flush=True)
    print(f"  Dataset: {dataset_root}", flush=True)
    print(f"{'='*60}\n", flush=True)
    logger.info("Deepfake classifier training (15 epochs, GPU optimized)")

    raw = discover_dataset_samples(dataset_root, max_per_class=max_per_class)
    if len(raw) < 4:
        logger.error("Not enough samples to train.")
        return

    labels_only = [l for _, l in raw]
    tr_idx, va_idx = stratified_indices(labels_only, val_ratio, seed=42)
    train_pairs = [raw[i] for i in tr_idx]
    val_pairs = [raw[i] for i in va_idx]

    train_pairs = balance_class_samples(train_pairs, rng=random.Random(41))

    train_ds_base = SampleListDataset(train_pairs, IMG_SIZE, train=True)
    val_ds = SampleListDataset(val_pairs, IMG_SIZE, train=False)
    indexed_train = IndexedWrapper(train_ds_base)
    num_train = len(indexed_train)

    sample_weights = np.ones(num_train, dtype=np.float64)
    for i in range(num_train):
        _, lab = train_ds_base.samples[i]
        sample_weights[i] = fake_sampler_boost if lab == LABEL_FAKE else 1.0
    sample_weights = np.maximum(sample_weights, 1e-4)

    weight_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weight_tensor,
        num_samples=num_train,
        replacement=True,
    )

    _nw = 8 if use_amp else 0
    _pm = use_amp
    _pw = _nw > 0
    train_loader = DataLoader(
        indexed_train,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=_nw,
        pin_memory=_pm,
        drop_last=True,
        persistent_workers=_pw,
        prefetch_factor=2 if _pw else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_nw,
        pin_memory=_pm,
        persistent_workers=_pw,
        prefetch_factor=2 if _pw else None,
    )



    model = DeepfakeClassifier(pretrained=True).to(device)
    criterion = FocalLossDeepfake(
        gamma=2.0,
        alpha_fake=0.55,
        label_smoothing=0.04,
        fn_soft_scale=1.0,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler = GradScaler("cuda", enabled=use_amp)

    best_score = -1.0
    best_fake_recall = 0.0
    recall_stall = 0
    collapse_count = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_fake_recall": [],
        "val_acc": [],
    }

    total_batches = len(train_loader)
    print(f"Training samples: {num_train}  |  Validation samples: {len(val_ds)}", flush=True)
    print(f"Batches per epoch: {total_batches}\n", flush=True)

    t0 = time.time()
    interrupted = False
    try:
      for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        seen = 0
        epoch_t0 = time.time()
        print(f"--- Epoch {epoch+1}/{epochs} ---", flush=True)

        for batch_idx, (images, labels, idxs) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1).float()
            idxs = idxs.cpu().numpy()
            sw = torch.as_tensor(sample_weights[idxs], dtype=torch.float32, device=device).unsqueeze(1)
            sw = sw / (sw.mean() + 1e-8)

            lam_used = None
            if mixup_alpha > 0 and images.size(0) > 1:
                images, labels, sw, lam_used = mixup_batch(images, labels, sw, mixup_alpha, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels, sample_weight=sw)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            seen += images.size(0)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
                print(f"  batch {batch_idx+1}/{total_batches}  running_loss={train_loss/max(seen,1):.4f}", flush=True)

            if lam_used is None:
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    fn = (labels > 0.5) & (preds < 0.5)
                    if fn.any():
                        for j in fn.nonzero(as_tuple=True)[0]:
                            wi = int(idxs[j.item()])
                            sample_weights[wi] = min(sample_weights[wi] * hard_mine_mult, hard_mine_cap)
                    fp = (labels < 0.5) & (preds > 0.5)
                    if fp.any():
                        for j in fp.nonzero(as_tuple=True)[0]:
                            wi = int(idxs[j.item()])
                            sample_weights[wi] = min(sample_weights[wi] * 1.12, hard_mine_cap)

        scheduler.step()
        epoch_train_loss = train_loss / max(seen, 1)
        weight_tensor.copy_(torch.from_numpy(sample_weights))

        val_loss, val_fr, val_rr, val_acc = _eval_loader_metrics(model, val_loader, device, criterion)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["val_fake_recall"].append(val_fr)
        history["val_acc"].append(val_acc)

        score = val_fr * 0.72 + val_acc * 0.28
        elapsed = time.time() - epoch_t0
        print(
            f"  => Epoch {epoch+1}/{epochs} completed in {elapsed:.1f}s  "
            f"loss={epoch_train_loss:.4f}  "
            f"accuracy={val_acc:.4f}  "
            f"fake_recall={val_fr:.4f}  "
            f"real_accuracy={val_rr:.4f}",
            flush=True,
        )
        logger.info(
            f"Epoch {epoch+1}/{epochs} loss={epoch_train_loss:.4f} "
            f"acc={val_acc:.4f} fake_recall={val_fr:.4f} real_acc={val_rr:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), SAVE_DIR / "deepfake_model.pth")
            print(f"  ** Saved best model (score={score:.4f}) **", flush=True)
            logger.info(f"saved best model (score={score:.4f})")

        # Collapse detection: if model predicts all one class
        if val_fr < 0.05 or val_rr < 0.05:
            collapse_count += 1
            if collapse_count >= 3:
                new_lr = optimizer.param_groups[0]['lr'] * 0.5
                for pg in optimizer.param_groups:
                    pg['lr'] = new_lr
                print(f"  !! Collapse detected ({collapse_count}x), lr reset to {new_lr:.2e}", flush=True)
                collapse_count = 0
        else:
            collapse_count = 0

        # Early stopping if fake recall stalls
        if val_fr > best_fake_recall:
            best_fake_recall = val_fr
            recall_stall = 0
        else:
            recall_stall += 1
            if recall_stall >= 5 and epoch >= 7:
                print(f"  !! Fake recall stalled for {recall_stall} epochs, stopping early.", flush=True)
                logger.info(f"early stop: fake recall stalled at {best_fake_recall:.4f}")
                break

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[Ctrl+C] Training interrupted at epoch {epoch+1}/{epochs}", flush=True)
        print(f"  Saving current model state...", flush=True)
        torch.save(model.state_dict(), SAVE_DIR / "deepfake_model.pth")
        print(f"  Model saved to: {SAVE_DIR / 'deepfake_model.pth'}", flush=True)
        logger.info(f"interrupted at epoch {epoch+1}, model saved")

    total_min = (time.time() - t0) / 60
    print(f"\n{'='*60}", flush=True)
    print(f"  Training complete in {total_min:.1f} min  |  Best score: {best_score:.4f}", flush=True)
    print(f"  Model saved to: {SAVE_DIR / 'deepfake_model.pth'}", flush=True)
    print(f"{'='*60}\n", flush=True)
    logger.info(f"done in {total_min:.1f} min best_score={best_score:.4f}")
    _plot_history(history)


if __name__ == "__main__":
    try:
        train(batch_size=48)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            print("\n[OOM] batch_size=48 caused CUDA OOM, retrying with batch_size=32...\n", flush=True)
            train(batch_size=32)
        else:
            raise
