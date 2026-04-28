import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class FFTBranch(nn.Module):
    def __init__(self, out_features: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class NoiseResidualBranch(nn.Module):
    def __init__(self, out_features: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def compute_fft_magnitude(images: torch.Tensor) -> torch.Tensor:
    gray = images.mean(dim=1, keepdim=True)
    fft = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shift)
    log_mag = torch.log1p(magnitude)
    B = log_mag.size(0)
    for i in range(B):
        min_val = log_mag[i].min()
        max_val = log_mag[i].max()
        if max_val > min_val:
            log_mag[i] = (log_mag[i] - min_val) / (max_val - min_val)
    return log_mag


def compute_noise_residual(images: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    sigma = kernel_size / 6.0
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel_1d = g / g.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel = kernel_2d.repeat(3, 1, 1, 1).to(images.device)
    padding = kernel_size // 2
    blurred = F.conv2d(images, kernel, padding=padding, groups=3)
    return images - blurred


class DeepfakeClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.55):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.rgb_features = backbone.features
        self.rgb_avgpool = backbone.avgpool
        rgb_dim = 768
        self.fft_branch = FFTBranch(out_features=256)
        self.noise_branch = NoiseResidualBranch(out_features=256)
        fused_dim = rgb_dim + 256 + 256
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.85),
            nn.Linear(256, 1),
        )
        nn.init.constant_(self.classifier[-1].bias, 0.0)

    def forward(
        self,
        rgb: torch.Tensor,
        fft_input: torch.Tensor = None,
        noise_input: torch.Tensor = None,
    ) -> torch.Tensor:
        if fft_input is None:
            fft_input = compute_fft_magnitude(rgb)
        if noise_input is None:
            noise_input = compute_noise_residual(rgb)
        rgb_feat = self.rgb_features(rgb)
        rgb_feat = self.rgb_avgpool(rgb_feat)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)
        rgb_feat = F.normalize(rgb_feat, dim=1, eps=1e-6)
        fft_feat = F.normalize(self.fft_branch(fft_input), dim=1, eps=1e-6)
        noise_feat = F.normalize(self.noise_branch(noise_input), dim=1, eps=1e-6)
        fused = torch.cat([rgb_feat, fft_feat, noise_feat], dim=1)
        return self.classifier(fused)


class DeepfakeClassifierV1(nn.Module):
    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.rgb_features = backbone.features
        self.rgb_avgpool = backbone.avgpool
        rgb_dim = 768
        self.fft_branch = FFTBranch(out_features=256)
        self.noise_branch = NoiseResidualBranch(out_features=256)
        fused_dim = rgb_dim + 256 + 256
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        fft_input: torch.Tensor = None,
        noise_input: torch.Tensor = None,
    ) -> torch.Tensor:
        if fft_input is None:
            fft_input = compute_fft_magnitude(rgb)
        if noise_input is None:
            noise_input = compute_noise_residual(rgb)
        rgb_feat = self.rgb_features(rgb)
        rgb_feat = self.rgb_avgpool(rgb_feat)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)
        fft_feat = self.fft_branch(fft_input)
        noise_feat = self.noise_branch(noise_input)
        fused = torch.cat([rgb_feat, fft_feat, noise_feat], dim=1)
        return self.classifier(fused)


class LegacyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import efficientnet_b0

        self.backbone = efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 1),
        )

    def forward(self, x, **kwargs):
        return self.backbone(x)


def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeClassifier(pretrained=False).to(device)
    dummy_rgb = torch.randn(2, 3, 256, 256, device=device)
    out = model(dummy_rgb)
    print(f"Model output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,}")


if __name__ == "__main__":
    test_model()
