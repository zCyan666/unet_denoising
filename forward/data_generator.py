import os
import random
from typing import Mapping, Sequence, MutableMapping, Any
import json
import harmonica as hm
import numpy as np


def sphere_magnetic_field(x, y, z, x0, y0, z0, radius, susceptibility, inc_f, dec_f, B0):
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




def cube_magnetic_field(x: np.ndarray, y: np.ndarray, z: np.ndarray, length: float, width: float, height: float,
                        cx: float, cy: float, cz: float, susceptibility: float, inc_f: int, dec_f: int, B0: int):
    """
    生成随机位置和尺寸的立方体磁异常
    """
    x1, x2 = cx - length / 2, cx + length / 2
    y1, y2 = cy - width / 2, cy + width / 2
    z1, z2 = cz, cz + height

    # 计算磁化强度 (A/m)
    mu0 = 4 * np.pi * 1e-7
    H0 = B0 * 1e-9 / mu0
    J = susceptibility * H0

    inc_rad = np.radians(inc_f)
    dec_rad = np.radians(dec_f)
    Mx = J * np.cos(inc_rad) * np.cos(dec_rad)
    My = J * np.cos(inc_rad) * np.sin(dec_rad)
    Mz = J * np.sin(inc_rad)

    # 计算磁场三分量
    b_e, b_n, b_u = hm.prism_magnetic(
        [x.ravel(), y.ravel(), z.ravel()],
        [x1, x2, y1, y2, z1, z2],
        (Mx, My, Mz),
        field="b"
    )

    b_e = b_e.reshape(x.shape)
    b_n = b_n.reshape(x.shape)
    b_u = b_u.reshape(x.shape)

    # 投影到地磁场方向（单位向量）
    inc_rad = np.radians(inc_f)
    dec_rad = np.radians(dec_f)
    fx = np.cos(inc_rad) * np.cos(dec_rad)
    fy = np.cos(inc_rad) * np.sin(dec_rad)
    fz = np.sin(inc_rad)

    anomaly = b_e * fx + b_n * fy + b_u * fz

    return anomaly


def generate_multi_sphere_anomaly(grid_size: int, area_size: int, susceptibility: float | np.ndarray, inc_f: int,
                                  dec_f: int, n_spheres: int, B0: int):
    """
    生成多个随机球体的磁异常

    参数:
        grid_size: 网格大小
        area_size: 区域大小 (米)
        n_spheres: 球体数量
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

    # 初始化总异常
    anomaly = np.zeros_like(X, dtype=float)
    spheres_info: MutableMapping[str, Any] = {'shape': 'sphere'}
    for i in range(n_spheres):
        # 随机球体参数
        radius = np.random.uniform(5, 20)
        bound = (area_size - radius - np.random.uniform(5, 10))
        cx = np.random.uniform(-bound, bound)
        cy = np.random.uniform(-bound, bound)
        cz = np.random.uniform(40, 60)  # 球心埋深

        kappa = susceptibility if isinstance(susceptibility, float) else susceptibility[i]

        # 计算当前球体的磁异常
        sphere_anom = sphere_magnetic_field(
            X, Y, Z, cx, cy, cz, radius, kappa, inc_f, dec_f, B0
        )

        anomaly += sphere_anom

        # 保存球体信息
        spheres_info.update(
            {
                'ID: ' + str(i + 1):
                {
                    'location': [cx, cy, cz],
                    'radius': radius,
                    'susceptibility': kappa,
                }
            }
        )

    return anomaly, spheres_info

def generate_multi_cube_anomaly(grid_size: int, area_size: int, susceptibility: float | np.ndarray, inc_f: int,
                                  dec_f: int, n_cubes: int, B0: int):
    x = np.linspace(-area_size, area_size, grid_size)
    y = np.linspace(-area_size, area_size, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # 地表观测

    anomaly = np.zeros_like(X, dtype=float)
    cubes_info: MutableMapping[str, Any] = {'shape': 'cube'}
    for i in range(n_cubes):

        # 随机长方体参数
        length = np.random.uniform(20, 40)  # x 方向长度
        width = np.random.uniform(20, 40)  # y 方向长度
        height = np.random.uniform(20, 40)  # z 方向厚度

        cx = np.random.uniform(-area_size / 2, area_size / 2)  # 中心 x
        cy = np.random.uniform(-area_size / 2, area_size / 2)  # 中心 y
        cz = np.random.uniform(20, 50)  # 顶面埋深

        kappa = susceptibility if isinstance(susceptibility, float) else susceptibility[i]
        cube_ano = cube_magnetic_field(X, Y, Z, length, width, height, cx, cy, cz,
                                            kappa, inc_f, dec_f, B0)
        anomaly += cube_ano
        cubes_info.update(
            {
                'ID: ' + str(i + 1):
                {
                    'location': [cx, cy, cz],
                    'length': length,
                    'width': width,
                    'height': height,
                    'susceptibility': kappa
                }
            }
        )

    return anomaly, cubes_info

def generate_plate_anomaly(grid_size: int, area_size: int, susceptibility: float, inc_f: int, dec_f: int, B0: int):
    """
    使用地磁场总强度 B0 生成板状体磁异常
    """
    x = np.linspace(-area_size, area_size, grid_size)
    y = np.linspace(-area_size, area_size, grid_size)
    x, y = np.meshgrid(x, y)

    # 随机生成板状体参数
    b = np.random.uniform(5, 10)  # 半宽度（m）
    h = np.random.uniform(10, 20)  # 顶面埋深（m）
    D = np.random.uniform(50, 200)  # 向下延深（m）
    alpha = 30
    # kappa = np.random.uniform(0.3, 0.8)  # 磁化率（SI）

    # 随机位置
    margin = b + area_size
    x0 = np.random.uniform(-area_size + margin, area_size - margin)
    y0 = np.random.uniform(-area_size + margin, area_size - margin)

    # 随机走向
    strike_angle = np.random.uniform(0, 180)
    # 计算磁异常
    # 常数
    mu0 = 4 * np.pi * 1e-7
    C = 1e-7  # μ₀/4π

    # 计算磁化强度 J (A/m)
    # H0 = B0 / μ₀，但 B0 是 nT，需要转换为 T
    # B0_T = B0 * 1e-9  # nT -> T
    H0 = B0 / mu0  # A/m
    J = susceptibility * H0  # A/m

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
    anomaly = Hax * np.cos(inc_rad) + Za * np.sin(inc_rad)

    params = {
        'shape': 'plate',
        'ID: 1':
        {
            'location': [x0, y0],
            'strike_angle': strike_angle,
            'b': b, 'h': h, 'D': D, 'alpha': alpha
        }
    }

    return anomaly, params


def add_mixed_noise(data, gaussian_ratio=0.2, pepper_ratio=0.2, salt_ratio=0.2):
    """
    添加混合噪声（参数更简洁），不归一化

    参数:
        data: 磁异常数据（nT）
        snr_db: 信噪比（分贝）
        salt_pepper_density: 椒盐噪声总密度
        salt_ratio: 椒噪声占比（剩余为盐噪声）
    """
    noisy = data.copy()

    # 1. 添加高斯噪声
    gaussian_std = np.std(noisy) * gaussian_ratio
    gaussian = np.random.normal(0, gaussian_std, noisy.shape)
    noisy = noisy + gaussian

    # 2. 添加椒盐噪声
    data_prob = 1 - pepper_ratio - salt_ratio
    mask = np.random.choice((0, -1, 1), size=data.shape[:2], p=[pepper_ratio, data_prob, salt_ratio])

    noisy[mask == 1] = np.max(data)
    noisy[mask == 0] = np.min(data)

    return noisy


# standardize = lambda x: (x - np.mean(x)) / np.std(x)
def standardize_min_max(x, y):
    max_xy = max(np.max(x), np.max(y))
    min_xy = min(np.min(x), np.min(y))
    x_hat = (x - min_xy) / (max_xy - min_xy)
    y_hat = (y - min_xy) / (max_xy - min_xy)
    return y_hat, x_hat


def random_float_range(min_v, max_v):
    return min_v + random.random() * (max_v - min_v)


def random_float_range_numpy(min_v, max_v, size=None):
    return min_v + np.random.random(size) * (max_v - min_v)


def restore_value(x, x_orign_min, x_orign_max):
    return x * (x_orign_max - x_orign_min) + x_orign_min


def combine_anomaly(*, grid_size, area_size, inc_f, dec_f, n_spheres, n_cubes):
    # 总磁场强度常量
    B0 = 50000

    # 定义各个物体磁化率
    spheres_sus = random_float_range_numpy(0.05, 0.3, n_spheres)
    plate_sus = random_float_range(0.1, 0.2)
    cube_sus = random_float_range_numpy(0.05, 0.2, n_cubes)

    spheres_ano, sp_info = generate_multi_sphere_anomaly(grid_size, area_size, spheres_sus, inc_f, dec_f, n_spheres, B0)
    plate_ano, plate_info = generate_plate_anomaly(grid_size, area_size, plate_sus, inc_f, dec_f, B0)
    cube_ano, cube_info = generate_multi_cube_anomaly(grid_size, area_size, cube_sus, inc_f, dec_f, n_cubes, B0)

    total_ano = spheres_ano + plate_ano + cube_ano
    return total_ano, (sp_info, plate_info, cube_info)


def create_training_pairs(i: int, save_dir: str) -> tuple[int, tuple[Mapping, ...]]:
    n_spheres = random.randint(1, 4)
    n_cubes = random.randint(1, 3)

    gaussian_ratio = random_float_range(0.01, 0.4)
    salt_ratio = random_float_range(0.0, 0.1)
    pepper_ratio = random_float_range(0.0, 0.1)

    anomaly, groups = combine_anomaly(
        grid_size=160, area_size=80, inc_f=90, dec_f=0, n_spheres=n_spheres, n_cubes=n_cubes
    )

    anomaly_noise = add_mixed_noise(anomaly, gaussian_ratio, pepper_ratio, salt_ratio)

    ano_train, ano_label = standardize_min_max(anomaly, anomaly_noise)

    np.save(os.path.join(save_dir, f"anomaly_{i}_{n_spheres}s_{n_cubes}c.npy"), ano_train)
    np.save(os.path.join(save_dir, f"anomaly_mask_{i}_{n_spheres}s_{n_cubes}c.npy"), ano_label)

    return i, groups

def generate_datasets(num_sample: int, save_dir: str, saving_info: bool):
    from concurrent.futures import ProcessPoolExecutor as P
    import functools
    creator = functools.partial(create_training_pairs, save_dir=save_dir)
    with P(max_workers=os.cpu_count()) as p:
        objs = list(p.map(creator, range(num_sample)))

    if not saving_info:
        return

    if len(objs) > 0:
        objs.sort(key=lambda x: x[0])
        parent_path = os.path.dirname(save_dir)
        with open(os.path.join(parent_path, "anomaly_info.json"), "w") as f:
            json.dump(objs, f, indent=2)

# ============ 主程序 ============
if __name__ == "__main__":
    directory_train = "../data/mag_train"
    directory_test = "../data/mag_test"

    os.makedirs(directory_train, exist_ok=True)
    os.makedirs(directory_test, exist_ok=True)

    # create_training_pairs(0, directory_test, None)
    generate_datasets(10000, directory_train, False)
    generate_datasets(10, directory_test, True)
