import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# ==============================================================================
# 1. 基础配置与物理常数
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 2. 数据加载与预处理
# ==============================================================================
def load_data(filepath):
    """读取 CSV 文件并进行插值处理"""
    try:
        df = pd.read_csv(filepath, sep=None, engine='python')
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"读取失败: {e}")
        return None, None, None

    # 准备全波段波长 (0.3 um 到 25 um)
    target_wl = np.linspace(0.3, 25.0, 3000)
    
    # 插值函数
    f_n = interp1d(df['wl'], df['n'], kind='linear', fill_value="extrapolate")
    f_k = interp1d(df['wl'], df['k'], kind='linear', fill_value="extrapolate")
    
    n_arr = f_n(target_wl)
    k_arr = f_k(target_wl)
    
    # 物理修正 k < 0
    k_arr[k_arr < 0] = 1e-9
    
    # 数据补全
    if df['wl'].min() > 0.3:
        idx_start = np.searchsorted(target_wl, df['wl'].min())
        n_arr[:idx_start] = n_arr[idx_start]
        k_arr[:idx_start] = k_arr[idx_start]
        
    return target_wl, n_arr, k_arr

# ==============================================================================
# 3. 光学模型定义
# ==============================================================================
def model_beer_lambert(wl, n, k, d_um):
    """比尔-朗伯定律模型 (几何光学近似)"""
    alpha = 4 * np.pi * k / wl
    # 垂直入射表面反射率近似
    R_surf = ((n - 1) / (n + 1))**2
    # 假设底部全反射，光程加倍
    emissivity = (1 - R_surf) * (1 - np.exp(-alpha * 2 * d_um))
    return emissivity

def model_tmm(wl_arr, n_arr, k_arr, d_um):
    """传输矩阵法模型 (波动光学)"""
    emissivity_list = []
    
    # 简化的银基底 (Ag)
    n_sub = 0.05
    k_sub = 4.0
    eta_sub = n_sub - 1j * k_sub
    eta_0 = 1.0
    
    for i, lam in enumerate(wl_arr):
        n = n_arr[i]
        k = k_arr[i]
        N_film = n - 1j * k
        
        # 相位厚度
        delta = 2 * np.pi * N_film * d_um / lam
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        eta_film = N_film
        
        if abs(eta_film) < 1e-10: eta_film = 1e-10
            
        M11 = cos_d
        M12 = (1j / eta_film) * sin_d
        M21 = 1j * eta_film * sin_d
        M22 = cos_d
        
        # 系统矩阵
        B = M11 * 1 + M12 * eta_sub
        C = M21 * 1 + M22 * eta_sub
        
        # 反射系数 r
        r = (eta_0 * B - C) / (eta_0 * B + C)
        R = np.abs(r)**2
        
        # 发射率 = 1 - R (假设无透射)
        emissivity_list.append(1 - R)
        
    return np.array(emissivity_list)

# ==============================================================================
# 4. 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    print(">>> 正在运行 Q1: PDMS 薄膜发射率分析...")
    
    file_path = 'pdms_data.csv'
    if not os.path.exists(file_path):
        print(f"错误: 未找到 {file_path}")
    else:
        wl_axis, n_vals, k_vals = load_data(file_path)
        
        # --- 绘图 1: 模型对比 ---
        d_compare = 50 # um
        E_beer = model_beer_lambert(wl_axis, n_vals, k_vals, d_compare)
        E_tmm = model_tmm(wl_axis, n_vals, k_vals, d_compare)

        plt.figure(figsize=(10, 6))
        plt.plot(wl_axis, E_beer, 'g--', label='Model 1: Beer-Lambert (Geometric)', linewidth=1.5, alpha=0.8)
        plt.plot(wl_axis, E_tmm, 'r-', label='Model 2: Transfer Matrix (Wave Optic)', linewidth=1.0)
        plt.axvspan(8, 13, color='cyan', alpha=0.15, label='Atmospheric Window (8-13 $\mu m$)')
        plt.title(f'Model Comparison: Emissivity of {d_compare} $\mu m$ PDMS Film', fontsize=14)
        plt.xlabel('Wavelength ($\mu m$)', fontsize=12)
        plt.ylabel('Emissivity', fontsize=12)
        plt.xlim(0.3, 25)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.show()

        # --- 绘图 2: 厚度影响 ---
        thicknesses = [10, 30, 50, 100, 200]
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(thicknesses)))

        for i, d in enumerate(thicknesses):
            E_curve = model_tmm(wl_axis, n_vals, k_vals, d)
            ax2.plot(wl_axis, E_curve, label=f'Thickness = {d} $\mu m$', color=colors[i], linewidth=1.2)

        ax2.axvspan(8, 13, color='gray', alpha=0.2, label='Atmospheric Window')
        ax2.set_title('Spectral Emissivity of PDMS Film with Varying Thickness', fontsize=16)
        ax2.set_xlabel('Wavelength ($\mu m$)', fontsize=14)
        ax2.set_ylabel('Emissivity', fontsize=14)
        ax2.set_xlim(0.3, 20)
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc='center right')
        plt.tight_layout()
        plt.show()