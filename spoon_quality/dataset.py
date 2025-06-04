import os
from glob import glob
from typing import Callable, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import remove_black_background


class SpoonDataset(Dataset):
    """Dataset para cargar imágenes de cucharas.

    Las imágenes se redimensionan a 1024x1024 y se devuelven como tensores RGBA.
    Si `preprocess` es True, se elimina el fondo negro y se guarda con
    transparencia en una carpeta temporal.
    """

    def __init__(self, folder: str, preprocess: bool = False):
        self.files = sorted(glob(os.path.join(folder, '*')))
        self.preprocess = preprocess
        self.tf = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        if preprocess:
            self._tmp_dir = os.path.join(folder, '.tmp_no_bg')
            os.makedirs(self._tmp_dir, exist_ok=True)
        else:
            self._tmp_dir = None

    def __len__(self) -> int:
        return len(self.files)

    def _get_path(self, index: int) -> str:
        path = self.files[index]
        if not self.preprocess:
            return path
        base = os.path.basename(path)
        out_path = os.path.join(self._tmp_dir, os.path.splitext(base)[0] + '.png')
        if not os.path.exists(out_path):
            remove_black_background(path, out_path)
        return out_path

    def __getitem__(self, index: int):
        path = self._get_path(index)
        img = Image.open(path).convert('RGBA')
        return self.tf(img), path

