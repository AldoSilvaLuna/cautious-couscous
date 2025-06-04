import os
import cv2
import numpy as np
from PIL import Image
import torch


def remove_black_background(input_path: str, output_path: str, black_tol: int = 30) -> None:
    """Quita el fondo negro de una imagen y guarda PNG con transparencia."""
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f'No se pudo leer la imagen: {input_path}')
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([179, 255, black_tol])
    mask_black = cv2.inRange(img_hsv, lower, upper)
    mask_fg = cv2.bitwise_not(mask_black)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img_rgb)
    alpha = mask_fg
    img_rgba = cv2.merge([r, g, b, alpha])
    img_pil = Image.fromarray(img_rgba)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_pil.save(output_path, format='PNG')


def visualize_features(model, dataloader, device='cpu', num_images=4, out_dir='visualizations'):
    """Guarda reconstrucciones o mapas de activación para entender el aprendizaje."""
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for xb, paths in dataloader:
            xb = xb.to(device)
            if hasattr(model, 'dec'):
                xr = model(xb)
                xb = xb.cpu()
                xr = xr.cpu()
                for i in range(min(num_images, xb.size(0))):
                    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                    ax[0].imshow(xb[i].permute(1, 2, 0))
                    ax[0].axis('off')
                    ax[0].set_title('Entrada')
                    ax[1].imshow(xr[i].permute(1, 2, 0))
                    ax[1].axis('off')
                    ax[1].set_title('Reconstrucción')
                    name = os.path.splitext(os.path.basename(paths[i]))[0]
                    plt.savefig(os.path.join(out_dir, f'{name}.png'), bbox_inches='tight')
                    plt.close(fig)
            else:
                out = model.features(xb)
                feat = out.cpu()[0, 0].numpy()
                plt.imshow(feat, cmap='viridis')
                name = os.path.splitext(os.path.basename(paths[0]))[0]
                plt.axis('off')
                plt.savefig(os.path.join(out_dir, f'{name}_feat.png'), bbox_inches='tight')
                plt.close()
            break

