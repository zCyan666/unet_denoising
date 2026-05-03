import asyncio
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import functools
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False


def infinite_strike_plate_magnetic_with_B0(
        x, y, x0, y0, strike_angle, b, h, D, alpha,
        kappa, inc_f, dec_f, B0=50000
):
    """
    无限走向板状体磁异常（使用地磁场总强度 B0 和磁化率 kappa）

    参数:
        x, y: 观测点坐标网格（m）
        x0, y0: 板状体中心坐标（m）
        strike_angle: 走向角度（度）
        b: 板状体半宽度（m）
        h: 顶面埋深（m）
        D: 向下延深（m）
        alpha: 板状体倾角（度）
        kappa: 磁化率（SI，无量纲）
        inc_f, dec_f: 地磁场倾角和偏角（度）
        B0: 地磁场总强度（nT），默认 50000 nT

    返回:
        delta_T: 总磁场异常（nT）
    """
    # 常数
    mu0 = 4 * np.pi * 1e-7
    C = 1e-7  # μ₀/4π

    # 计算磁化强度 J (A/m)
    # H0 = B0 / μ₀，但 B0 是 nT，需要转换为 T
    # B0_T = B0 * 1e-9  # nT -> T
    H0 = B0 / mu0  # A/m
    J = kappa * H0  # A/m

    # 角度转弧度
    strike_rad = np.radians(strike_angle)
    alpha_rad = np.radians(alpha)
    inc_rad = np.radians(inc_f)

    # 有效磁化倾角
    gamma = inc_f - alpha
    gamma_rad = np.radians(gamma)

    # 坐标系变换
    cos_s = np.cos(strike_rad)
    sin_s = np.sin(strike_rad)

    dx = x - x0
    dy = y - y0
    u = dx * cos_s + dy * sin_s  # 垂直走向方向

    # 计算垂直走向方向的磁异常剖面
    x_prime = u * np.cos(alpha_rad)

    # 顶面贡献
    x1 = x_prime + b
    x2 = x_prime - b
    h_prime = h

    za_top = 2 * C * J * np.sin(gamma_rad) * (
            np.arctan2(x1, h_prime) - np.arctan2(x2, h_prime)
    )

    # 底面贡献
    h_bottom = h + D * np.sin(alpha_rad)
    x1_b = x_prime + b
    x2_b = x_prime - b

    za_bottom = 2 * C * J * np.sin(gamma_rad) * (
            np.arctan2(x1_b, h_bottom) - np.arctan2(x2_b, h_bottom)
    )

    Za = za_top - za_bottom

    # 水平分量
    hax_top = C * J * np.sin(gamma_rad) * np.log(
        (x1 ** 2 + h_prime ** 2) / (x2 ** 2 + h_prime ** 2)
    )
    hax_bottom = C * J * np.sin(gamma_rad) * np.log(
        (x1_b ** 2 + h_bottom ** 2) / (x2_b ** 2 + h_bottom ** 2)
    )
    Hax = hax_top - hax_bottom

    # 总场异常（已经是 nT，因为前面已转换）
    delta_T = Hax * np.cos(inc_rad) + Za * np.sin(inc_rad)

    return delta_T


def generate_plate_with_B0(grid_size=128, area_size=5000, inc_f=45, dec_f=45):
    """
    使用地磁场总强度 B0 生成板状体磁异常
    """
    x = np.linspace(-area_size, area_size, grid_size)
    y = np.linspace(-area_size, area_size, grid_size)
    X, Y = np.meshgrid(x, y)

    # 随机生成板状体参数
    b = np.random.uniform(0.1, 1)  # 半宽度（m）
    h = np.random.uniform(10, 20)  # 顶面埋深（m）
    D = np.random.uniform(400, 800)  # 向下延深（m）
    alpha = 30
    kappa = np.random.uniform(0.3, 0.8)  # 磁化率（SI）

    # 随机位置
    margin = b + 100
    x0 = np.random.uniform(-area_size + margin, area_size - margin)
    y0 = np.random.uniform(-area_size + margin, area_size - margin)

    # 随机走向
    strike_angle = np.random.uniform(0, 180)

    # 地磁场参数
    #inc_f = np.random.uniform(30, 60)
    #dec_f = np.random.uniform(-15, 20)
    B0 = 50000  # 地磁场总强度 (nT)

    # 计算磁异常
    anomaly = infinite_strike_plate_magnetic_with_B0(
        X, Y, x0, y0, strike_angle, b, h, D, alpha,
        kappa, inc_f, dec_f, B0
    )

    params = {
        'x0': x0, 'y0': y0,
        'strike_angle': strike_angle,
        'b': b, 'h': h, 'D': D, 'alpha': alpha,
        'kappa': kappa, 'inc_f': inc_f, 'dec_f': dec_f, 'B0': B0,
        'type': 'infinite_strike_plate'
    }

    return anomaly, X, Y, params

def sphere_magnetic_field(x, y, z, x0, y0, z0, radius, susceptibility, inc_f, dec_f, B0=50000):
    """
    计算单个球体的磁异常（偶极子近似）

    参数:
        x, y, z: 观测点坐标
        x0, y0, z0: 球心坐标 (米)
        radius: 球体半径 (米)
        susceptibility: 磁化率 (SI)
        inc_f, dec_f: 地磁场倾角和偏角 (度)
        B0: 地磁场总强度 (nT)

    返回:
        delta_T: 总磁场异常 (nT)
    """
    # 转换为 numpy 数组
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    original_shape = x.shape
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # 地磁场方向单位向量
    inc_f_rad = np.radians(inc_f)
    dec_f_rad = np.radians(dec_f)
    fx = np.cos(inc_f_rad) * np.cos(dec_f_rad)
    fy = np.cos(inc_f_rad) * np.sin(dec_f_rad)
    fz = np.sin(inc_f_rad)

    # 磁化强度
    mu0 = 4 * np.pi * 1e-7
    H0 = B0 * 1e-9 / mu0
    M_intensity = susceptibility * H0

    # 球体体积和磁矩
    volume = 4 / 3 * np.pi * radius ** 3
    moment = M_intensity * volume

    # 磁化方向单位向量
    mx, my, mz = fx, fy, fz  # 感应磁化，方向与地磁场一致

    # 常数
    C = 1e-7

    # 初始化结果
    delta_T = np.zeros(len(x))

    for i in range(len(x)):
        dx = x[i] - x0
        dy = y[i] - y0
        dz = z[i] - z0
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if r < 1e-6:
            continue

        r3 = r ** 3
        r5 = r ** 5

        dot_m_r = mx * dx + my * dy + mz * dz

        Bx = C * (3 * dx * dot_m_r / r5 - mx / r3) * moment
        By = C * (3 * dy * dot_m_r / r5 - my / r3) * moment
        Bz = C * (3 * dz * dot_m_r / r5 - mz / r3) * moment

        delta_T[i] = (Bx * fx + By * fy + Bz * fz) * 1e9

    return delta_T.reshape(original_shape)



def generate_multi_sphere_anomaly(grid_size=64, area_size=5000, inc_f=90,
                                  dec_f=90, n_spheres=None):
    """
    生成多个随机球体的磁异常

    参数:
        grid_size: 网格大小
        area_size: 区域大小 (米)
        n_spheres: 球体数量 (None 表示随机 1-5 个)

    返回:
        anomaly: 总磁异常
        X, Y: 坐标网格
        spheres_info: 所有球体的参数列表
    """
    # 观测网格
    x = np.linspace(-area_size, area_size, grid_size)
    y = np.linspace(-area_size, area_size, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # 随机决定球体数量
    if n_spheres is None:
        n_spheres = np.random.randint(1, 4)  # 1-5 个球体

    # 初始化总异常
    anomaly = np.zeros_like(X, dtype=float)
    spheres_info = []
    for i in range(n_spheres):
        # 随机球体参数
        radius = np.random.uniform(20, 30)  # 半径 50-200m
        bound = (area_size - radius - np.random.uniform(5, 10))
        cx = np.random.uniform(-bound, bound)
        cy = np.random.uniform(-bound, bound)

        cz = np.random.uniform(40, 60)  # 球心埋深
        susceptibility = 0.075  # 磁化率

        # 计算当前球体的磁异常
        sphere_anom = sphere_magnetic_field(
            X, Y, Z, cx, cy, cz, radius, susceptibility, inc_f, dec_f
        )

        anomaly += sphere_anom

        # 保存球体信息

        spheres_info.append({
            'id': i + 1,
            'cx': cx, 'cy': cy, 'cz': cz,
            'radius': radius,
            'susceptibility': susceptibility,
            'inc_f': inc_f,
            'dec_f': dec_f
        })

    return anomaly, X, Y, spheres_info, inc_f, dec_f


def add_mixed_noise_raw(data,
                        gaussian_std=5.0,
                        salt_prob=0.01,
                        pepper_prob=0.01):
    """
    直接添加混合噪声，不进行归一化

    参数:
        data: 2D numpy array，磁异常数据（单位 nT），如范围 -500 到 500
        gaussian_std: 高斯噪声标准差（nT），建议 1-10
        salt_prob: 椒噪声概率（高值异常）
        pepper_prob: 盐噪声概率（低值异常）

    返回:
        noisy_data: 添加噪声后的数据
    """
    noisy = data.copy()

    # 获取数据的实际最大最小值
    max_val = np.max(data)
    min_val = np.min(data)

    # 1. 添加高斯噪声
    gaussian_noise = np.random.normal(0, gaussian_std, noisy.shape)
    noisy = noisy + gaussian_noise

    # 2. 添加椒盐噪声（直接使用原始数据的极值）
    salt_mask = np.random.random(noisy.shape) < salt_prob
    pepper_mask = np.random.random(noisy.shape) < pepper_prob

    # 避免同一位置同时被覆盖
    salt_mask = salt_mask & ~pepper_mask
    pepper_mask = pepper_mask & ~salt_mask

    noisy[salt_mask] = max_val  # 椒噪声：数据最大值
    noisy[pepper_mask] = min_val  # 盐噪声：数据最小值

    return noisy

def add_noise(anomaly, noise_percentage=0.6):
    """添加高斯噪声"""
    signal_std = np.std(anomaly)
    noise_std = signal_std * noise_percentage
    noise = np.random.normal(0, noise_std, anomaly.shape)
    return anomaly + noise, noise


# standardize = lambda x: (x - np.mean(x)) / np.std(x)
def standardize_min_max(x, y):
    max_xy = max(np.max(x), np.max(y))
    min_xy = min(np.min(x), np.min(y))
    x_hat = (x - min_xy) / (max_xy - min_xy)
    y_hat = (y - min_xy) / (max_xy - min_xy)
    return y_hat, x_hat

def random_float_range(min_v, max_v):
    return min_v + random.random() * (max_v - min_v)

def generate_training_pair(grid_size=128, area_size=5000, inc_f=90,
                           dec_f=90, n_spheres=None, noise_percentage=None):
    """
    生成一对训练数据（多个球体）

    返回:
        clean: 干净磁异常
        noisy: 含噪磁异常 (如果 snr_db 不为 None)
        label: 标签
        X, Y: 坐标
        spheres_info: 球体信息
    """
    clean1, X, Y, spheres_info, inc_f, dec_f = generate_multi_sphere_anomaly(
        grid_size=grid_size, area_size=area_size, n_spheres=n_spheres, inc_f=inc_f, dec_f=dec_f
    )
    clean2, X, Y, params = generate_plate_with_B0(grid_size, area_size, inc_f, dec_f)

    clean = clean1 + clean2

    noisy = add_mixed_noise_custom(clean)
    return clean, noisy, X, Y, spheres_info

def create_anomaly(i, save_dir):
    noise_percentage = random_float_range(0.5, 0.9)
    n_sphere = random.randint(1, 4)

    clean, noisy, X, Y, spheres_info = generate_training_pair(
        grid_size=160, area_size=80, inc_f=90, dec_f=90,
        n_spheres=n_sphere, noise_percentage=noise_percentage
    )

    noisy_norm, clean_norm = standardize_min_max(clean, noisy)

    np.save(os.path.join(save_dir, f"anmoly_{i}.npy"), noisy_norm)
    np.save(os.path.join(save_dir, f"anmoly_mask_{i}.npy"), clean_norm)


def generate_datasets(num_sample, save_dir: str):
    creator = functools.partial(create_anomaly, save_dir=save_dir)
    with multiprocessing.Pool(processes=min(os.cpu_count(), 63)) as p:
        p.map(creator, range(num_sample))


# ============ 主程序 ============
if __name__ == "__main__":
    directory_train = "../data/mag_train_1"
    directory_test = "../data/mag_test_1"

    os.makedirs(directory_train, exist_ok=True)
    os.makedirs(directory_test, exist_ok=True)

    generate_datasets(10000, directory_train)
    generate_datasets(100, directory_test)
