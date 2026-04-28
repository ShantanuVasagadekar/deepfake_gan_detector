"""
gan_model.py — DCGAN Generator and Discriminator for deepfake detection.

Architecture follows the DCGAN paper (Radford et al., 2015) adapted for
128×128 RGB images.

After adversarial training the **Discriminator** is exported as the deepfake
detection model — its output represents P(real).
"""

import torch
import torch.nn as nn

# Latent vector dimensionality
NZ = 100
# Feature map sizes
NGF = 64  # Generator
NDF = 64  # Discriminator
# Image channels
NC = 3


# ═══════════════════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """
    DCGAN Generator: maps a latent vector z (NZ×1×1) → RGB image (3×128×128).

    Architecture (128×128 output):
        z → ConvT 4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128
    """

    def __init__(self, nz: int = NZ, ngf: int = NGF, nc: int = NC):
        super().__init__()
        self.main = nn.Sequential(
            # Input: nz × 1 × 1 → ngf*16 × 4 × 4
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            # → ngf*8 × 8 × 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # → ngf*4 × 16 × 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # → ngf*2 × 32 × 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # → ngf × 64 × 64
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # → nc × 128 × 128
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


# ═══════════════════════════════════════════════════════════════════════════
# Discriminator (doubles as the deepfake detector)
# ═══════════════════════════════════════════════════════════════════════════

class Discriminator(nn.Module):
    """
    DCGAN Discriminator: RGB image (3×128×128) → scalar P(real).

    Architecture (128×128 input):
        img → Conv 64×64 → 32×32 → 16×16 → 8×8 → 4×4 → 1×1
    """

    def __init__(self, nc: int = NC, ndf: int = NDF):
        super().__init__()
        self.main = nn.Sequential(
            # Input: nc × 128 × 128 → ndf × 64 × 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # → ndf*2 × 32 × 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # → ndf*4 × 16 × 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # → ndf*8 × 8 × 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # → ndf*16 × 4 × 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # → 1 × 1 × 1
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.main(img)


# ═══════════════════════════════════════════════════════════════════════════
# Quick sanity check
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(device)
    D = Discriminator().to(device)

    z = torch.randn(4, NZ, 1, 1, device=device)
    fake_imgs = G(z)
    scores = D(fake_imgs)

    print(f"Generator  output shape : {fake_imgs.shape}")   # [4, 3, 128, 128]
    print(f"Discriminator output    : {scores.shape}")       # [4, 1, 1, 1]
    print(f"Sample scores           : {scores.view(-1).tolist()}")

    total_G = sum(p.numel() for p in G.parameters())
    total_D = sum(p.numel() for p in D.parameters())
    print(f"Generator  params       : {total_G:,}")
    print(f"Discriminator params    : {total_D:,}")
