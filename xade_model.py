"""
xade_model.py — XADE Deepfake Detector Model Architecture.

Uses torchvision's EfficientNet-B4 backbone with a custom classifier head.
Trained on 140k Real and Fake Faces (StyleGAN-generated).
Reports AUC-ROC of 0.9992.

Checkpoint structure (from best_model.pt):
    Top-level keys: epoch, model_state_dict, optimizer_state_dict,
                    best_val_loss, train_samples, class_names
    State dict prefix: model.features.* (backbone), model.classifier.* (head)
    Classifier layers indexed: 0=Dropout, 1=Linear(1792,512), 2=ReLU,
                               3=BatchNorm1d(512), 4=Dropout, 5=Linear(512,2)
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class XADEClassifier(nn.Module):
    """
    XADE Deepfake Detector.
    Architecture: EfficientNet-B4 backbone + custom 2-layer classifier head.
    Labels: 0 = Real, 1 = Fake
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Load torchvision EfficientNet-B4 (just the architecture, weights come from checkpoint)
        base = efficientnet_b4(weights=None)

        # Keep the feature extractor (Conv stem + MBConv blocks + final Conv)
        self.features = base.features

        # Adaptive pooling (same as torchvision default)
        self.avgpool = base.avgpool  # AdaptiveAvgPool2d(1)

        # Custom classifier head (matches checkpoint exactly):
        #   0: Dropout(0.5)
        #   1: Linear(1792, 512)
        #   2: ReLU
        #   3: BatchNorm1d(512)
        #   4: Dropout(0.4)
        #   5: Linear(512, 2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1792, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_xade_model(weights_path: str = None, device: str = "cpu") -> XADEClassifier:
    """
    Instantiate XADE model and optionally load checkpoint weights.

    Args:
        weights_path: Path to best_model.pt checkpoint.
        device: 'cpu' or 'cuda'.

    Returns:
        Loaded model in eval mode.
    """
    model = XADEClassifier()

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        # The checkpoint wraps the state dict under 'model_state_dict'
        state_dict = checkpoint["model_state_dict"]

        # Keys in checkpoint are prefixed with "model." (e.g. model.features.0.0.weight)
        # Our module names are "features.*" and "classifier.*" (no "model." prefix)
        # Strip the "model." prefix
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned[k[len("model."):]] = v
            else:
                cleaned[k] = v

        model.load_state_dict(cleaned, strict=True)
        print(f"[XADE] Loaded {len(cleaned)} parameters from checkpoint "
              f"(epoch {checkpoint.get('epoch', '?')}, "
              f"best_val_loss={checkpoint.get('best_val_loss', '?')})")

    model.to(device)
    model.eval()
    return model
