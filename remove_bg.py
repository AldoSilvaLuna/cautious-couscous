import os
import cv2
import numpy as np
from PIL import Image

def remove_black_background(input_path, output_path, black_tol=30):
    """
    Elimina el fondo negro de una imagen y guarda el resultado con fondo transparente.
    
    - input_path: ruta al archivo original (JPG o PNG con fondo negro).
    - output_path: ruta donde se guardará el PNG resultante con transparencia.
    - black_tol: tolerancia para considerar un píxel como “fondo negro” (0 = negro puro,
                 valores más altos permitirán una gama más amplia de tonos oscuros).
    """

    # 1) Leer la imagen original con OpenCV en BGR
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {input_path}")

    # 2) Convertir a HSV para detectar tonos oscuros de forma más robusta
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 3) Crear máscara de píxeles que estén “cerca de negro”:
    #    - Tomamos V (valor/brillo) menor que black_tol => prácticamente negro
    #    - (Opcional) También podemos mirar S para descartar pixeles muy grises o similares,
    #      pero en la mayoría de fondos negros esto basta.
    lower = np.array([0, 0, 0])
    upper = np.array([179, 255, black_tol])
    mask_black = cv2.inRange(img_hsv, lower, upper)

    # Invertimos la máscara para que la cuchara (píxeles no-oscuros) quede en blanco (255)
    mask_fg = cv2.bitwise_not(mask_black)

    # 4) Crear la imagen RGBA de salida:
    #    - Convertimos BGR a RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    #    - Canales RGB por separado
    r, g, b = cv2.split(img_rgb)

    #    - El canal alfa: los píxeles “foreground” (parte de la cuchara) serán 255,
    #      y los de fondo (negro) 0 (transparentes)
    alpha = mask_fg

    # 5) Unir canales en una imagen con transparencia (RGBA)
    img_rgba = cv2.merge([r, g, b, alpha])

    # 6) Convertir a PIL para guardar en PNG (mantener transparencia)
    img_pil = Image.fromarray(img_rgba)

    # Asegurarse de que la carpeta de salida existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 7) Guardar como PNG con canal alfa
    img_pil.save(output_path, format="PNG")


if __name__ == "__main__":
    # Directorio donde están las imágenes originales con fondo negro
    input_folder = "datasets/test/DEFECTIVE"       # Cambiar por la ruta real
    # Carpeta de salida donde guardaremos los PNGs sin fondo
    output_folder = "datasets/no_bg/DEFECTIVE"

    # Crear carpeta de salida (si no existe)
    os.makedirs(output_folder, exist_ok=True)

    # Recorrer todos los archivos JPG/PNG en el directorio de entrada
    for filename in os.listdir(input_folder):
        base, ext = os.path.splitext(filename.lower())
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue

        input_path = os.path.join(input_folder, filename)
        # Queremos guardar como PNG (para mantener transparencia)
        output_filename = base + ".png"
        output_path = os.path.join(output_folder, output_filename)

        try:
            remove_black_background(input_path, output_path, black_tol=60)
            print(f"[OK] Procesado: {filename} → {output_filename}")
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
