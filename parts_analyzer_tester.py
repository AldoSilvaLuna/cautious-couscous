import os
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import cv2

image_size = 1024
# Threshold: 0.002129
# Threshold: 0.001235
# Threshold: 0.000357

# -------------------------------------------------------------------
# 1. Vuelve a definir tu dataset y el autoencoder (igual que antes)
# -------------------------------------------------------------------

class NormalDataset(Dataset):
    def __init__(self, folder, img_size=128):
        self.files = glob(os.path.join(folder, "*.png"))
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGBA")
        return self.tf(img), self.files[i]

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1),  # 128→64 # cambia 3→4 canales de entrada
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 64→32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 32→16
            nn.ReLU(True),
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #16→32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), #32→64
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 4, 3, stride=2, padding=1, output_padding=1),  #64→128
            nn.Sigmoid(),  # salida en [0,1]
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

# -------------------------------------------------------------------
# 2. Función para calcular umbral (opcional si ya lo calculaste)
# -------------------------------------------------------------------

def compute_threshold(model, dataloader, device="cpu"):
    model.to(device)
    model.eval()
    errors = []
    criterion = nn.MSELoss(reduction="none")
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            xr = model(xb)
            # Calcular MSE por pixel y luego promedio por ejemplo:
            per_pixel = criterion(xr, xb)            # shape (B,3,128,128)
            per_image = per_pixel.mean(dim=[1,2,3])  # shape (B,)
            batch_err = per_image.cpu().numpy()
            errors.extend(batch_err)
    errors = np.array(errors)
    thresh = errors.mean() + 3*errors.std()
    print(f"[compute_threshold] Threshold = {thresh:.6f}")
    return thresh

# -------------------------------------------------------------------
# 3. Función de inferencia: carga modelo y detecta anomalía en 1 imagen
# -------------------------------------------------------------------

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

def is_defective_old(model, img_path, threshold, img_size=128, device="cpu"):
    print("is_defective img_path", img_path)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    model.to(device)
    model.eval()
    img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)  # shape (1,3,128,128)
    with torch.no_grad():
        recon = model(img)
        err = ((recon - img)**2).mean().item()
    return err > threshold, err

def visualizar_reconstrucciones(model, dataloader, device="cpu", num_ejemplos=4, output_dir="reconstrucciones"):
    """
    Igual que antes, pero en lugar de plt.show() guarda cada par
    original/reconstrucción en archivos PNG dentro de output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for xb, rutas in dataloader:
            xb = xb.to(device)
            xr = model(xb)
            xb = xb.cpu()
            xr = xr.cpu()
            for i in range(min(num_ejemplos, xb.size(0))):
                fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                # Original
                axs[0].imshow(xb[i].permute(1, 2, 0).numpy())
                axs[0].set_title("Original")
                axs[0].axis("off")
                # Reconstrucción
                axs[1].imshow(xr[i].permute(1, 2, 0).numpy())
                axs[1].set_title("Reconstrucción")
                axs[1].axis("off")

                # Guardar el par en disco
                fname = os.path.basename(rutas[i])  # nombre original del archivo
                print("fname", fname)
                savepath = os.path.join(output_dir, f"recon_{fname}")
                print("savepath", savepath)
                plt.savefig(savepath, bbox_inches="tight")
                plt.close(fig)
            break  # solo el primer batch
    print(f"Reconstrucciones guardadas en la carpeta: {output_dir}")



# ----------------------------------------------------
# Función para visualizar mapa de errores (heatmap)
# ----------------------------------------------------

def visualize_error_map(model, img_path, img_size=1024, device="cpu", output_dir="heatmaps"):
    os.makedirs(output_dir, exist_ok=True)
    # Cargar imagen como RGBA y transformarla
    img = Image.open(img_path).convert("RGBA")
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    xb = tf(img).unsqueeze(0).to(device)  # (1,4,H,W)

    model.eval()
    with torch.no_grad():
        xr = model(xb)  # (1,4,H,W)
    xb_cpu = xb.cpu()[0].permute(1, 2, 0).numpy()  # (H,W,4)
    xr_cpu = xr.cpu()[0].permute(1, 2, 0).numpy()  # (H,W,4)

    # Calcular error por canal
    error_tensor = (xr_cpu - xb_cpu) ** 2  # (H,W,4)
    # Error promedio sobre canales para obtener heatmap (H,W)
    error_map = np.mean(error_tensor, axis=2)
    # Normalizar a [0,1]
    error_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)

    # Crear heatmap usando colormap de Matplotlib
    heatmap = cv2.applyColorMap((error_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superponer heatmap con la imagen original RGB
    rgb_orig = xb_cpu[:, :, :3]
    rgb_orig_disp = (rgb_orig * 255).astype(np.uint8)
    overlay = (0.6 * rgb_orig_disp + 0.4 * heatmap).astype(np.uint8)

    # Guardar imágenes: original RGBA, reconstrucción RGBA, heatmap y overlay
    base = os.path.splitext(os.path.basename(img_path))[0]
    # Original (mantenemos transparencia en PNG)
    Image.fromarray((xb_cpu * 255).astype(np.uint8)).save(os.path.join(output_dir, f"orig_{base}.png"))
    # Reconstrucción
    Image.fromarray((xr_cpu * 255).astype(np.uint8)).save(os.path.join(output_dir, f"recon_{base}.png"))
    # Heatmap solo
    plt.imsave(os.path.join(output_dir, f"heatmap_{base}.png"), error_norm, cmap='jet')
    # Overlay heatmap sobre original RGB
    Image.fromarray(overlay).save(os.path.join(output_dir, f"overlay_{base}.png"))

    print(f"Generado heatmap para {img_path}, guardado en {output_dir}")

# -------------------------------------------------------------------
# 4. Bloque principal: cargar pesos y hacer inferencias
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 4.1. Define tu dispositivo: CPU si no tienes GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4.2. Ruta a tu modelo guardado
    MODEL_PATH = "autoencoder.pth"
    
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró {MODEL_PATH}. Asegúrate de entrenar primero.")

    # 4.3. Crea la instancia del modelo y carga los pesos
    model = ConvAutoencoder().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print("[INFO] Modelo cargado correctamente desde", MODEL_PATH)

    # 4.4. Si deseas recalcular el umbral a partir de un conjunto de validación:
    #     Solo descomenta estas líneas y ajusta DATA_FOLDER_VALIDATION a tu carpeta.
    # DATA_FOLDER_VALIDATION = "datasets/validacion"  # punto a tus imágenes de validación
    # batch_size = 16
    # ds_val = NormalDataset(DATA_FOLDER_VALIDATION, img_size=128)
    # dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4)
    # threshold = compute_threshold(model, dl_val, device=device)

    # (B) Preparar DataLoader de validación solo con cucharas buenas:
    VALIDATION_FOLDER = "datasets/no_bg/DEFECTIVE"
    ds_val = NormalDataset(VALIDATION_FOLDER, img_size=image_size)
    dl_val = DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=6, pin_memory=True)

    # (C) Visualizar reconstrucciones para “ver” qué está aprendiendo la red
    print("Visualizando reconstrucciones en el set de validación…")
    visualizar_reconstrucciones(model, dl_val, device=device, num_ejemplos=6)


    # 4.5. Si ya calculaste el umbral antes y lo guardaste manualmente, asigna:
    threshold = 0.002465  # <— reemplaza por el valor que calculaste (mean + 3·std)

    # 4.6. Prueba de inferencia sobre algunas imágenes nuevas
    # TEST_IMAGES = [
    #     "datasets/test/OK/BURST_20250602_161710_4.jpg",
    #     "datasets/test/DEFECTIVE/BURST_20250602_164838_5.jpg",
    #     "datasets/test/DEFECTIVE/BURST_20250602_164905_15.jpg",
    #     "datasets/test/DEFECTIVE/BURST_20250602_164906_22.jpg",
    #     "datasets/test/DEFECTIVE/BURST_20250602_164907_25.jpg",
    #     "datasets/test/DEFECTIVE/BURST_20250602_164909_32.jpg",
    #     # Agrega todas las rutas que quieras probar
    #     "datasets/no_bg/OK/burst_20250602_161718_37.png",
    #     "datasets/no_bg/DEFECTIVE/burst_20250602_164840_15.png",
    #     "datasets/no_bg/DEFECTIVE/burst_20250602_164909_32.png"
    # ]
    TEST_IMAGES = glob(os.path.join('datasets/no_bg/DEFECTIVE', "*.png"))

    for img_path in TEST_IMAGES:
        if not os.path.isfile(img_path):
            print(f"[WARN] No existe {img_path}, se omite.")
            continue
        salida, error = is_defective(model, img_path, threshold, img_size=image_size, device=device)
        etiqueta = "DEFECTUOSA" if salida else "PERFECTA"
        print(f"{img_path} → {etiqueta} (error={error:.6f})")
        # Generar heatmap para cada imagen de prueba
        visualize_error_map(model, img_path, img_size=image_size, device=device)
