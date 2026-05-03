import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
import re
import time
import sys
from typing import Optional, TextIO, Generator
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
# from models.network_resnet import Residual
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2 import Transform

from loss_utils import WeightL1Loss, MSEWithKLLoss
from models.network_unet import (
    UnetDenoiser,
    UnetPlusPlusDenoise
)
from plots.plots_helper import show_image

def train_test_splitter_classification(folder, split_amount=0.1, random=123, shuffle=True):
    files = os.listdir(folder)

    def key_fn(filename):
        return [int(c) for c in re.split(r'(\d+)', filename) if c.isdigit()]

    files.sort(key=key_fn)

    inputs = tuple(os.path.join(folder, f) for f in files if "mask" not in f)
    targets = tuple(os.path.join(folder, f) for f in files if f.rfind("mask") != -1)

    X = np.array(inputs)
    y = np.array(targets)

    assert X.shape == y.shape, "You don't have same nums"

    train_features, test_features, train_masks, test_masks = train_test_split(X, y,
                                                                              test_size=split_amount,
                                                                              random_state=random,
                                                                              shuffle=shuffle
                                                                              )
    return (train_features, train_masks), (test_features, test_masks)


class MyClassificationDataSet(Dataset):
    def __init__(self, files: list | tuple, transform: Optional[Transform]):
        self.inputs, self.masks = files
        self.transform = transform
        # self.inputs = self.inputs[:16]
        # self.masks = self.masks[:16]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        img = Image.open(self.inputs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        img = F.to_image(img)
        mask = F.to_image(mask)

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        mask = (mask > 0).float()

        return img, mask


class MyClassificationDataSetNumpy(Dataset):
    def __init__(self, files: list | tuple, transform: Optional[Transform]):
        self.inputs, self.masks = files
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        mag = torch.tensor(np.load(self.inputs[idx]), dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(np.load(self.masks[idx]), dtype=torch.float32).unsqueeze(0)

        return mag, target


class MyNetworkTrainer:
    def __init__(self, pt_path, data_path, device, resize, batch_size, train_epoch,
                 net, dataset_cls, loss_fn, optim, *, scaler=None, scheduler=None, view_plot=True,
                 split_ranges=0.1, random_state=12345, shuffle=True, multi_processing=False):
        self.device = device
        self.train_epoch = train_epoch
        self.save_pth = Path(pt_path) if isinstance(pt_path, str) else pt_path

        self.train_transform = v2.Compose([
            v2.CenterCrop(resize),
            # v2.Resize(resize // 2),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToDtype(dtype=torch.float32, scale=True),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = v2.Compose([
            v2.CenterCrop(resize),
            # v2.Resize(resize // 2),
            v2.ToDtype(dtype=torch.float32, scale=True)
        ])

        self.net = net.to(self.device)
        self.loss_fn = loss_fn
        self.optim = optim
        self.scheduler = scheduler
        train_files, val_files = train_test_splitter_classification(data_path, split_ranges, random_state,
                                                                    shuffle)

        train_datasets = dataset_cls(train_files, self.train_transform)
        val_datasets = dataset_cls(val_files, self.test_transform)

        self.train_loader = DataLoader(
            train_datasets,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if self.device == "cuda" else False,
            num_workers=os.cpu_count() if multi_processing else 0,
        )

        self.val_loader = DataLoader(
            val_datasets,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if self.device == "cuda" else False,
            num_workers=os.cpu_count() if multi_processing else 0,
        )
        self.batch_size = batch_size
        self.scaler = scaler
        self.view_plot = view_plot

    @staticmethod
    def check_training_health(loss, epoch, thres):
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"Loss is NaN/Inf at epoch {epoch}")

        if loss > thres:
            raise OverflowError(f"Loss is too high: {loss.item()}")

    def _train_one_epoch(self, progress, epoch, thres):
        self.net.train()
        total_loss = 0

        for idx, (imgs, masks) in enumerate(progress):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            with torch.amp.autocast(device_type=self.device):
                logits = self.net(imgs)
                loss = self.loss_fn(logits, masks)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            self.check_training_health(loss, epoch, thres)

            total_loss += loss.item()

            progress.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def _validation(self):
        self.net.eval()
        total_loss = 0
        total_psnr_metric = 0

        trigger_once = True if self.view_plot else False

        with torch.no_grad():
            for imgs, masks in self.val_loader:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                logits = self.net(imgs)
                loss = self.loss_fn(logits, masks)

                if isinstance(logits, tuple):
                    logits = sum(logits) / len(logits)

                psnr = compute_psnr_metrics(masks, logits)

                if trigger_once:
                    idx = random.randint(0, self.batch_size - 1)
                    if self.device == 'cpu':
                        show_image([imgs[idx].clone(), masks[idx].clone(), logits[idx].clone()], cmap='gist_rainbow_r')
                    else:
                        show_image([imgs[idx].cpu(), masks[idx].cpu(), logits[idx].cpu()], cmap='gist_rainbow_r')
                    trigger_once = False

                total_loss += loss.item()
                total_psnr_metric += psnr.item()


        return total_loss / len(self.val_loader), total_psnr_metric / len(self.val_loader)

    def train(self, logger: TextIO) -> None:
        loss_curr = float('inf')
        try:
            for epoch in range(self.train_epoch):
                prog = tqdm.tqdm(self.train_loader, desc=f"Epoch: [{epoch + 1}/{self.train_epoch}]")

                train_loss = self._train_one_epoch(prog, epoch, 10.0)
                val_loss, psnr_metric = self._validation()

                logger.write(f"Epoch: {epoch}\n")
                logger.write(f"Train loss: {train_loss:.6f}\nVal loss: {val_loss:.6f}\n")
                logger.write(f"PSNR metric: {psnr_metric:.2f}\n")
                logger.flush()

                print('=' * 70)
                print(f"Epoch {epoch + 1}/{self.train_epoch}")
                print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                print(f"psnr metric: {psnr_metric:.6f}")

                if self.scheduler is not None:
                    self.scheduler.step()
                    print(f"Current Lr: {self.scheduler.get_last_lr()}")

                if val_loss < loss_curr:
                    loss_curr = val_loss
                    print(f"Saving model...")
                    torch.save(self.net.state_dict(), self.save_pth)
                print('=' * 70)
                time.sleep(0.01)
        except KeyboardInterrupt:
            logger.close()
            print('Keyboard interrupted, cleaning log file...', file=sys.stderr)
            if os.path.exists(logger.name):
                os.remove(logger.name)
            sys.exit(1)
        finally:
            if not logger.closed:
                logger.close()

def compute_psnr_metrics(original, denoised, max_val=1.0):
    mse = torch.mean((original - denoised) ** 2)
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr

def create_new_log(name: str, directory: str, suffix='.txt') -> TextIO:
    def gen_name() -> Generator:
        i = 0
        while True:
            yield f"{name}_{i}{suffix}"
            i += 1

    gen = gen_name()
    Path(directory).mkdir(exist_ok=True)
    for _ in filter(lambda s: s.endswith(suffix), os.listdir(directory)):
        next(gen)
    return open(os.path.join(directory, next(gen)), 'w', encoding='utf-8')

def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    os.makedirs("./checkpoints", exist_ok=True)
    RANDOM_SEED = 666
    DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
    BATCHSIZE = 16
    LR = 0.001
    EPOCH = 50
    IMGSIZE = 256
    GAMMA = 0.95

    seed_everything(RANDOM_SEED)

    # Get model
    config_path = './configs/net.yaml'
    model = UnetPlusPlusDenoise.from_yaml(config_path)
    # model = UnetDenoiser.from_yaml(config_path)
    pt_path = f'./checkpoints/{model.__class__.__name__}.pth'
    
    # Prepare loss function
    # loss_fn = nn.MSELoss()
    # loss_fn = MSEWithKLLoss(0.02)
    loss_fn = WeightL1Loss()

    # Training utils
    # 为了使用dropout，我们需要设置权重衰减
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(DEVICE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

    data_path = './data/mag_train'
    trainer = MyNetworkTrainer(
        pt_path,
        data_path,
        DEVICE,
        IMGSIZE,
        BATCHSIZE,
        EPOCH,
        model,
        MyClassificationDataSetNumpy,
        loss_fn,
        optimizer,
        scaler=scaler,
        scheduler=None,
        view_plot=False,
        random_state=RANDOM_SEED,
    )

    logger = create_new_log('train_log', './plots/logs')
    trainer.train(logger)


if __name__ == "__main__":
    main()
