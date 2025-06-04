import os
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# import matplotlib.pyplot as plt
# import numpy as np

# Threshold: 0.002129
# Threshold: 0.001235
# Threshold: 0.000320
# Threshold: 0.000341
# Threshold: 0.002465

image_size = 1024

# 1. Dataset para cargar solo imágenes "buenas"
class NormalDataset(Dataset):
    def __init__(self, folder, img_size=128):
        self.files = glob(os.path.join(folder, "*.png"))
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGBA")
        return self.tf(img), self.files[i]

# 2. Autoencoder simple
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: la resolución se divide entre 2 tres veces
        self.enc = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1),   # 1024 → 512
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 512  → 256
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 256  → 128
            nn.ReLU(True),
            # Opcional: otra capa para obtener un espacio latente más pequeño
            # nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128 → 64
            # nn.ReLU(True),
        )
        # Decoder: la resolución se restaura multiplicando por 2 tres veces
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 128 → 256
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 256 → 512
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 4, 3, stride=2, padding=1, output_padding=1),   # 512 → 1024
            nn.Sigmoid(),  # Normaliza salida a [0,1]
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)


# 3. Entrenamiento
def train_autoencoder(data_folder, epochs=20, batch_size=16, lr=1e-3, img_size=128, device="cuda"):
    print("train_autoencoder", data_folder)
    # DataLoader
    ds = NormalDataset(data_folder, img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Modelo, optimizador, criterio
    model = ConvAutoencoder().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Loop de entrenamiento
    model.train()
    for ep in range(1, epochs+1):
        running_loss = 0.0
        for xb, _ in dl:
            xb = xb.to(device)  # xb: (B, 4, img_size, img_size)
            xr = model(xb)      # xr: (B, 4, img_size, img_size)
            loss = criterion(xr, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(ds)
        print(f"Epoch {ep:02d}/{epochs}  Loss: {epoch_loss:.6f}")

    # Guardar pesos
    torch.save(model.state_dict(), "autoencoder.pth")
    return model, dl

# 4. Cálculo de umbral (mean + 3·std) sobre reconstrucción
def compute_threshold(model, dataloader, device="cuda"):
    model.eval()
    errors = []
    criterion = nn.MSELoss(reduction="none")
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)  # (B,4,H,W)
            xr = model(xb)      # (B,4,H,W)
            per_pixel = criterion(xr, xb)      # shape (B,4,H,W)
            per_image = per_pixel.mean(dim=[1,2,3]).cpu().numpy()  # error medio por imagen
            # MSE por imagen
            # batch_err = ((xr - xb)**2).mean(dim=[1,2,3]).cpu().numpy()
            errors.extend(per_image)
    errors = np.array(errors)
    thresh = errors.mean() + 3*errors.std()
    # print(f"Threshold: {thresh:.6f}")
    print(f"Threshold de anomalía (RGBA): {thresh:.6f}")
    return thresh

# 5. Inferencia: marcar como defectuosa si error > umbral
def is_defective(model, img_path, threshold, img_size=128, device="cpu"):
    # Carga directa en RGBA
    img = Image.open(img_path).convert("RGBA")
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()  # produce un tensor (4, H, W)
    ])
    xb = tf(img).unsqueeze(0).to(device)  # shape (1, 4, img_size, img_size)

    model.eval()
    with torch.no_grad():
        xr = model(xb)                    # shape (1, 4, img_size, img_size)
        err = ((xr - xb) ** 2).mean().item()
    return err > threshold, err
    
def is_defective_old(model, img_path, threshold, img_size=128, device="cuda"):
    print("is_defective img_size", img_size)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    model.eval()
    img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(img)
        err = ((recon - img)**2).mean().item()
    return err > threshold, err

if __name__ == "__main__":
    # Carpeta con ≥200 imágenes de piezas correctas
    DATA_FOLDER = "datasets/no_bg/OK"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Entrena autoencoder
    model, dl = train_autoencoder(DATA_FOLDER, epochs=25, batch_size=8, img_size=image_size, device=DEVICE)

    # 2) Calcula umbral de anomalía
    thresh = compute_threshold(model, dl, device=DEVICE)

    # 3) Prueba sobre una imagen nueva
    test_img = "datasets/no_bg/DEFECTIVE/burst_20250602_164908_30.png"
    defective, err = is_defective(model, test_img, thresh, img_size=image_size, device=DEVICE)
    print(f"{test_img} → {'DEFECTIVE' if defective else 'OK'}  (error={err:.6f})")
