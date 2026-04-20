import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Dict, Any, Optional
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib.pyplot as plt

from models.network_unet import (
    UnetDenoiser,
    UnetPlusPlusDenoise
)
import numpy as np
import torch
import pywt
import re
import random

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False


def wavelet_denoising(signal, wavelet='db4', level=4, method='soft', threshold_factor=1.0):
    """
    小波去噪函数

    参数:
        signal: 输入信号（1D或2D）
        wavelet: 小波基类型（'db4', 'sym4', 'coif4'等）
        level: 分解层数
        method: 阈值方法 ('soft'软阈值, 'hard'硬阈值)
        sigma: 噪声标准差（None则自动估计）

    返回:
        denoised: 去噪后的信号
        threshold: 使用的阈值
    """
    # 小波分解
    coeffs = pywt.wavedec2(signal, wavelet, level=level)

    # 估计噪声标准差（使用第一层高频系数）
    detail_coeffs = coeffs[-1]  # 最后一层的高频系数 (cH, cV, cD)
    sigma = np.median(np.abs(detail_coeffs[0])) / 0.6745

    # 计算通用阈值
    threshold = sigma * np.sqrt(2 * np.log(signal.size)) * threshold_factor

    # 阈值处理高频系数
    denoised_coeffs = [coeffs[0]]  # 保留低频系数

    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]

        if method == 'soft':
            cH_denoised = pywt.threshold(cH, threshold, mode='soft')
            cV_denoised = pywt.threshold(cV, threshold, mode='soft')
            cD_denoised = pywt.threshold(cD, threshold, mode='soft')
        else:
            cH_denoised = pywt.threshold(cH, threshold, mode='hard')
            cV_denoised = pywt.threshold(cV, threshold, mode='hard')
            cD_denoised = pywt.threshold(cD, threshold, mode='hard')

        denoised_coeffs.append((cH_denoised, cV_denoised, cD_denoised))

    # 小波重构
    denoised = pywt.waverec2(denoised_coeffs, wavelet)

    # 裁剪到原始尺寸
    denoised = denoised[:signal.shape[0], :signal.shape[1]]

    return denoised, threshold, sigma


def wavelet_denoise_2d(anomaly, wavelet='db4', level: Optional[int] = None, method='soft'):
    """2D磁异常小波去噪"""
    denoised, threshold, sigma = wavelet_denoising(anomaly, wavelet=wavelet, level=level, method=method)
    return denoised


def init_model(MyModel: Any, check_path: str | Path, device: str, **model_params):
    model = MyModel(**model_params)
    checkpoint = torch.load(check_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict(model, arr: str | np.ndarray, device, img_preprocess=None) -> torch.Tensor:
    if isinstance(arr, str):
        arr = np.load(arr)

    arr = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    pred = model(arr)
    #     pred = torch.clamp(pred, 0.0, 1.0)
    return pred

def montecarlo_dropout_predict(model, arr: str | np.ndarray, device):
    if isinstance(arr, str):
        arr = np.load(arr)


def plot_figure(*arr_args, **styles) -> None:
    _, axes = plt.subplots(1, len(arr_args))
    for i, arr in enumerate(arr_args):
        if isinstance(arr, torch.Tensor):
            arr = np.squeeze(arr.detach().cpu())
        im = axes[i].imshow(arr, **styles)
        plt.colorbar(im, ax=axes[i], fraction=0.1, aspect=10)
    plt.tight_layout()
    plt.show()


def compute_psnr_metrics(original, denoised, max_val=1.0):
    mse = np.mean((original - denoised) ** 2)
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr, mse


if __name__ == '__main__':

    DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
    hparams: Dict[str, Any] = {
        'in_channel': 1,
        'out_channel': 1,
        'network_depth': 5,
        'require_1x1_conv': True
    }

    directory = './Data/mag_test'
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
    model1 = init_model(UnetDenoiser, './checkpoints/unet_denoise_1.pth', DEVICE, **hparams)
    model2 = init_model(UnetPlusPlusDenoise, './checkpoints/unet++_denoise_1.pth', DEVICE, **hparams)

    pred1, pred2 = predict(model1, noisy, DEVICE), predict(model2, noisy, DEVICE)

    pred2 = sum(pred2) / len(pred2)

    pred1 = np.squeeze(pred1.detach().cpu().numpy())
    pred2 = np.squeeze(pred2.detach().cpu().numpy())

    # 小波变换磁异常
    denoised = wavelet_denoise_2d(noisy, wavelet='sym4', level=4)

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
    ax.plot(np.linspace(-80, 80, 160), pred1[noisy.shape[0] // 2, :], label='unet去噪')
    ax.plot(np.linspace(-80, 80, 160), pred2[noisy.shape[0] // 2, :], label='unet++去噪')
    ax.plot(np.linspace(-80, 80, 160), denoised[noisy.shape[0] // 2, :], label='db4去噪')
    ax.set_title('磁异常中心位置纵轴切片')
    ax.legend()
    ax.grid(True, alpha=0.5)

    plot_style = dict(cmap='gist_rainbow_r')
    plot_figure(noisy, target, denoised, pred1, pred2, **plot_style)
