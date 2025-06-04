from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def save_model(model: nn.Module, path: str) -> None:
    """Guarda los pesos del modelo en `path`."""
    torch.save(model.state_dict(), path)


def load_model(model_cls, path: str, device: str = 'cpu') -> nn.Module:
    """Carga los pesos de `path` en una instancia de `model_cls`."""
    model = model_cls().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


class Autoencoder(nn.Module):
    """Autoencoder convolucional sencillo para detección de anomalías."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 4, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)


class Classifier(nn.Module):
    """Red convolucional simple para clasificación binaria."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class UNet(nn.Module):
    """Implementación muy simplificada de U-Net."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottom = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(True),
        )
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(True),
        )
        self.out = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottom(p2)
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return torch.sigmoid(self.out(d1))

