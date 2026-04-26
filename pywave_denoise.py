import pywt
import numpy as np
def wavelet_denoise(signal, wavelet='db4', level=4, method='soft', threshold_factor=1.0):
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

    return denoised