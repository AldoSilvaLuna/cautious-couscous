# Cautious Couscous!!

## parts_analyzer_api_service.py

Upload images to directory

## parts_analyzer_trainer.py

Train model

## parts_analyzer_tester.py

Test model training

## remove_bg.py

Remove background from images

## spoon_quality

Paquete con utilidades para entrenar modelos de detección de defectos en cucharas. Incluye tres opciones de entrenamiento:

- **autoencoder**: aprende solo con cucharas correctas y detecta anomalías midiendo el error de reconstrucción.
- **classifier**: clasificador binario que distingue entre cucharas correctas y defectuosas.
- **segmenter**: red de segmentación que detecta el área de la imperfección en cucharas defectuosas (requiere máscaras de entrenamiento).

Todos los modelos trabajan con imágenes RGBA de 1024x1024 píxeles. Las funciones para guardar/cargar modelos y visualizar lo aprendido se encuentran en el paquete.

### Ejemplos de entrenamiento

Entrenamiento del autoencoder con una carpeta solo de cucharas correctas:

```bash
python3 -m spoon_quality.train autoencoder datasets/correct --epochs 20 --out modelos/autoencoder.pth
```

Entrenamiento del clasificador usando carpetas de imágenes correctas y defectuosas:

```bash
python3 -m spoon_quality.train classifier datasets/correct datasets/defects --epochs 20 --out modelos/classifier.pth
```

Entrenamiento del segmentador (las máscaras PNG deben estar en `datasets/defects/masks`):

```bash
python3 -m spoon_quality.train segmenter datasets/defects --epochs 20 --out modelos/segmenter.pth
```

### Ejemplos de prueba de modelos

Cargar un autoencoder y calcular el error de reconstrucción de una imagen:

```python
from PIL import Image
import torch
from torchvision import transforms
from spoon_quality.models import Autoencoder, load_model

model = load_model(Autoencoder, "modelos/autoencoder.pth")
tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
img = tf(Image.open("spoon.png").convert("RGBA")).unsqueeze(0)
with torch.no_grad():
    recon = model(img)
    mse = torch.mean((recon - img)**2).item()
print("Reconstruccion:", mse)
```

Cargar el clasificador para predecir si una cuchara es defectuosa:

```python
from PIL import Image
import torch
from torchvision import transforms
from spoon_quality.models import Classifier, load_model

model = load_model(Classifier, "modelos/classifier.pth")
tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
x = tf(Image.open("spoon.png").convert("RGBA")).unsqueeze(0)
with torch.no_grad():
    out = torch.softmax(model(x), dim=1)
    print("probabilidad defectuosa:", out[0,1].item())
```

Cargar el segmentador para obtener la máscara de imperfecciones:

```python
from PIL import Image
import torch
from torchvision import transforms
from spoon_quality.models import UNet, load_model

model = load_model(UNet, "modelos/segmenter.pth")
tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
x = tf(Image.open("defect_spoon.png").convert("RGBA")).unsqueeze(0)
with torch.no_grad():
    mask = model(x)[0,0]
Image.fromarray((mask.numpy()*255).astype("uint8")).save("mask.png")
```

