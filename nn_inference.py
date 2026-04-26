import os
from torch import nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Dict, Any, Optional, List, overload
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib.pyplot as plt

from pywave_denoise import *
from models.network_unet import (
    UnetDenoiser,
    UnetPlusPlusDenoise
)
import numpy as np
import torch

import re
import random

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

def compute_psnr_metrics(original, denoised, max_val=1.0):
    mse = np.mean((original - denoised) ** 2)
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr, mse

def init_model(MyModel: Any, check_path: str | Path, device: str, yaml_path: str | Path):
    model = MyModel.from_yaml(yaml_path)
    checkpoint = torch.load(check_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def predict(model, arr: str | np.ndarray, device):
    if isinstance(arr, str):
        arr = np.load(arr)

    arr = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        return model(arr)

def montecarlo_dropout_predict(model, arr: str | np.ndarray, device, max_it=30):
    for module in model.modules():
        if isinstance(module, (nn.Dropout2d, nn.Dropout)):
            module.train()

    def accumulate():
        for it in range(max_it):
            logits = predict(model, arr, device)
            res = sum(logits) / len(logits)
            predicted.append(res)

    predicted: List[torch.Tensor] = []
    accumulate()

    # 计算统计量
    predictions = torch.stack(predicted)  # (max_it, C, H, W)
    mean = predictions.mean(dim=0)  # 去噪结果
    std = predictions.std(dim=0)  # 不确定性

    return mean, std

def plot_figure(*arr_args, **styles) -> None:
    if len(arr_args) % 2 == 1:
        _, axes = plt.subplots(1, len(arr_args))
    else:
        _, axes = plt.subplots(2, len(arr_args) // 2)
    for i, arr in enumerate(arr_args):
        if isinstance(arr, torch.Tensor):
            arr = np.squeeze(arr.detach().cpu())
        if axes.ndim == 2:
            if i < axes.shape[-1]:
                im = axes[0][i].imshow(arr, **styles)
                plt.colorbar(im, ax=axes[0][i], fraction=0.1, aspect=10)
            else:
                im = axes[1][i - axes.shape[-1]].imshow(arr, **styles)
                plt.colorbar(im, ax=axes[1][i - axes.shape[-1]], fraction=0.1, aspect=10)
        else:
            im = axes[i].imshow(arr, **styles)
            plt.colorbar(im, ax=axes[i], fraction=0.1, aspect=10)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':

    DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'

    directory = './data/mag_test'
    yaml_path = './configs/net.yaml'

    files = os.listdir(directory)
    files.sort(key=lambda s: [int(t) for t in re.split(r'(\d+)', s) if t.isdigit()])

    pairs = []
    for i in range(0, len(files), 2):
        pairs.append(tuple(files[i: i + 2]))

    noisy, target = pairs[random.randint(0, len(files) // 2 - 1)]
    print(noisy, '|', target)

    # 原始含噪声磁异常
    noisy = np.load(os.path.join(directory, noisy))

    # 干净磁异常标签
    target = np.load(os.path.join(directory, target))

    # 预测的磁异常
    model1 = init_model(UnetDenoiser, './checkpoints/unet_denoise_1.pth', DEVICE, yaml_path)
    model2 = init_model(UnetPlusPlusDenoise, './checkpoints/unet++_denoise_1.pth', DEVICE, yaml_path)

    pred1 = predict(model1, noisy, DEVICE)
    pred2, std = montecarlo_dropout_predict(model2, noisy, DEVICE)

    pred1 = np.squeeze(pred1.detach().cpu().numpy())
    pred2 = np.squeeze(pred2.detach().cpu().numpy())

    # 小波变换磁异常
    denoised = wavelet_denoise(noisy, wavelet='db4', level=4)

    # snr_0, mse0 = compute_psnr_metrics(noisy, target)
    psnr_1 = peak_signal_noise_ratio(target, denoised, data_range=1.0)
    psnr_2 = peak_signal_noise_ratio(target, pred1, data_range=1.0)
    psnr_3 = peak_signal_noise_ratio(target, pred2, data_range=1.0)

    ssim1 = structural_similarity(target, denoised, win_size=11, data_range=1.0, channel_axis=None)
    ssim2 = structural_similarity(target, pred1, win_size=11, data_range=1.0, channel_axis=None)
    ssim3 = structural_similarity(target, pred2, win_size=11, data_range=1.0, channel_axis=None)

    # print(f'磁异常信噪比: {snr_0} db, 均方误差MSE: {mse0}')
    print(f'小波变换磁异常信噪比: {psnr_1:.3f} db, SSIM指标: {ssim1:.4f}')
    print(f'unet神经网络磁异常信噪比: {psnr_2:.3f} db, SSIM指标: {ssim2:.4f}')
    print(f'unet++神经网络磁异常信噪比: {psnr_3:.3f} db, SSIM指标: {ssim3:.4f}')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.linspace(-80, 80, 160), noisy[noisy.shape[0] // 2, :], alpha=0.8, ls='--', label='加噪声')
    ax.plot(np.linspace(-80, 80, 160), target[noisy.shape[0] // 2, :], label='原始磁异常')
    ax.plot(np.linspace(-80, 80, 160), pred1[noisy.shape[0] // 2, :], label='attention-unet去噪')
    ax.plot(np.linspace(-80, 80, 160), pred2[noisy.shape[0] // 2, :], label='unet++去噪')
    ax.plot(np.linspace(-80, 80, 160), denoised[noisy.shape[0] // 2, :], label='db4去噪')
    ax.set_title('磁异常中心位置纵轴切片')
    ax.legend()
    ax.grid(True, alpha=0.5)

    plot_style = dict(cmap='gist_rainbow_r')
    plot_figure(noisy, target, denoised, pred1, pred2, std, **plot_style)
