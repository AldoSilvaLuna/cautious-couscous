import argparse
import os
from typing import Tuple
from glob import glob
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import SpoonDataset
from .models import Autoencoder, Classifier, UNet, save_model
from .utils import visualize_features


def train_autoencoder(data_dir: str, epochs: int, device: str) -> Tuple[Autoencoder, DataLoader]:
    dataset = SpoonDataset(data_dir, preprocess=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    model = Autoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    for ep in range(1, epochs + 1):
        total = 0.0
        for xb, _ in loader:
            xb = xb.to(device)
            xr = model(xb)
            loss = criterion(xr, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f'Epoch {ep:02d} Loss {total/len(dataset):.6f}')
    return model, loader


def train_classifier(ok_dir: str, defect_dir: str, epochs: int, device: str) -> Classifier:
    ok_ds = SpoonDataset(ok_dir, preprocess=True)
    def_ds = SpoonDataset(defect_dir, preprocess=True)
    imgs = ok_ds.files + def_ds.files
    labels = [0]*len(ok_ds) + [1]*len(def_ds)

    class TempDS(torch.utils.data.Dataset):
        def __init__(self, imgs, labels):
            self.imgs = imgs
            self.labels = labels
            self.tf = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ])
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            img = Image.open(self.imgs[idx]).convert('RGBA')
            label = self.labels[idx]
            return self.tf(img), torch.tensor(label, dtype=torch.long)

    ds = TempDS(imgs, labels)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    model = Classifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f'Epoch {ep:02d} Loss {total/len(ds):.6f}')
    return model


def train_segmenter(defect_dir: str, epochs: int, device: str) -> UNet:
    mask_dir = os.path.join(defect_dir, 'masks')
    imgs = sorted(glob(os.path.join(defect_dir, '*.png')))
    masks = [os.path.join(mask_dir, os.path.basename(p)) for p in imgs]

    class SegDS(torch.utils.data.Dataset):
        def __init__(self, imgs, masks):
            self.imgs = imgs
            self.masks = masks
            self.tf_img = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ])
            self.tf_mask = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ])
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            img = Image.open(self.imgs[idx]).convert('RGBA')
            mask = Image.open(self.masks[idx]).convert('L')
            return self.tf_img(img), self.tf_mask(mask)

    ds = SegDS(imgs, masks)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2)
    model = UNet(num_classes=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    for ep in range(1, epochs + 1):
        total = 0.0
        for xb, mb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            pred = model(xb)
            loss = criterion(pred, mb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f'Epoch {ep:02d} Loss {total/len(ds):.6f}')
    return model


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento de modelos de cucharas')
    sub = parser.add_subparsers(dest='mode', required=True)

    auto_p = sub.add_parser('autoencoder')
    auto_p.add_argument('data', help='Carpeta de cucharas buenas')
    auto_p.add_argument('--epochs', type=int, default=10)
    auto_p.add_argument('--out', default='autoencoder.pth')

    cls_p = sub.add_parser('classifier')
    cls_p.add_argument('ok_dir')
    cls_p.add_argument('def_dir')
    cls_p.add_argument('--epochs', type=int, default=10)
    cls_p.add_argument('--out', default='classifier.pth')

    seg_p = sub.add_parser('segmenter')
    seg_p.add_argument('def_dir')
    seg_p.add_argument('--epochs', type=int, default=10)
    seg_p.add_argument('--out', default='segmenter.pth')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.mode == 'autoencoder':
        model, loader = train_autoencoder(args.data, args.epochs, device)
        save_model(model, args.out)
        visualize_features(model, loader, device=device)
    elif args.mode == 'classifier':
        model = train_classifier(args.ok_dir, args.def_dir, args.epochs, device)
        save_model(model, args.out)
    elif args.mode == 'segmenter':
        model = train_segmenter(args.def_dir, args.epochs, device)
        save_model(model, args.out)


if __name__ == '__main__':
    main()

