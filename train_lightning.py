import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import amp
from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np
from pathlib import Path
from PIL import Image

# Re-use existing components
from classifier_model import DeepfakeClassifier
from face_preprocessing import (
    discover_dataset_samples,
    balance_class_samples,
    stratified_indices,
    SampleListDataset,
    LABEL_FAKE,
)

# Configuration for H100 Cloud Node
IMG_SIZE = 256
BATCH_SIZE = 128  # High VRAM on H100 allows much larger batches
NUM_WORKERS = 16   # High CPU core count on cloud
LR = 4e-4         # Linear scaling of LR for larger batches
WEIGHT_DECAY = 1.2e-4
EPOCHS = 15

class FocalLossDeepfake(nn.Module):
    def __init__(self, gamma=2.0, alpha_fake=0.55, label_smoothing=0.04, fn_soft_scale=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha_fake = alpha_fake
        self.alpha_real = 1.0 - alpha_fake
        self.label_smoothing = label_smoothing
        self.fn_soft_scale = fn_soft_scale

    def forward(self, logits, targets, sample_weight=None):
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

class DeepfakeLightningModule(L.LightningModule):
    def __init__(self, lr=LR, weight_decay=WEIGHT_DECAY):
        super().__init__()
        self.save_hyperparameters()
        self.model = DeepfakeClassifier(pretrained=True)
        self.criterion = FocalLossDeepfake()
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.unsqueeze(1).float()
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.unsqueeze(1).float()
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        # Recall metrics
        fake_mask = labels > 0.5
        n_fake = fake_mask.sum().clamp(min=1.0)
        fake_rec = ((preds > 0.5) & fake_mask).sum() / n_fake
        
        n_real = (~fake_mask).sum().clamp(min=1.0)
        real_ok = ((preds < 0.5) & ~fake_mask).sum() / n_real
        
        acc = (preds == labels).float().mean()
        
        metrics = {"val_loss": loss, "val_acc": acc, "val_fake_recall": fake_rec, "val_real_acc": real_ok}
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in self.validation_step_outputs]).mean()
        avg_recall = torch.stack([x["val_fake_recall"] for x in self.validation_step_outputs]).mean()
        avg_real_acc = torch.stack([x["val_real_acc"] for x in self.validation_step_outputs]).mean()
        
        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", avg_acc, prog_bar=True, sync_dist=True)
        self.log("val_fake_recall", avg_recall, prog_bar=True, sync_dist=True)
        self.log("val_real_acc", avg_real_acc, prog_bar=True, sync_dist=True)
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class DeepfakeDataModule(L.LightningDataModule):
    def __init__(self, data_root, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        raw = discover_dataset_samples(self.data_root)
        labels_only = [l for _, l in raw]
        tr_idx, va_idx = stratified_indices(labels_only, 0.2, seed=42)
        
        train_pairs = [raw[i] for i in tr_idx]
        val_pairs = [raw[i] for i in va_idx]
        
        # Balance training
        train_pairs = balance_class_samples(train_pairs)
        
        self.train_ds = SampleListDataset(train_pairs, IMG_SIZE, train=True)
        self.val_ds = SampleListDataset(val_pairs, IMG_SIZE, train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

if __name__ == "__main__":
    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    
    # 1. Setup Data & Model
    data_dir = str(Path(__file__).resolve().parent / "dataset")
    dm = DeepfakeDataModule(data_dir)
    model = DeepfakeLightningModule()

    # 2. Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="saved_models",
        filename="best_cloud_model",
        monitor="val_fake_recall",
        mode="max",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # 3. Trainer (Multi-GPU Optimized)
    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",  # Uses all available H100s
        strategy="ddp_find_unused_parameters_false",
        precision="bf16-mixed",  # Optimized for H100
        max_epochs=EPOCHS,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)
    
    # Save the final .pth for detect_image.py inference
    trainer.model.model.to("cpu")
    torch.save(trainer.model.model.state_dict(), "saved_models/deepfake_model.pth")
    print("\nCloud training complete. Model saved to saved_models/deepfake_model.pth")
