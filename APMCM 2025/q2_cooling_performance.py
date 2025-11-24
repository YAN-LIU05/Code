import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# ==============================================================================
# 1. 数据与物理模型
# ==============================================================================
wl_axis = np.linspace(0.3, 25.0, 3000)

def load_material_from_csv(filename, material_name):
    """读取CSV材料数据"""
    print(f"正在读取 {material_name} 数据: {filename} ...")
    if not os.path.exists(filename):
        # 只有在找不到文件时返回模拟数据以防崩溃
        print(f"警告: 未找到 {filename}，使用模拟数据。")
        return np.ones_like(wl_axis)*1.5, np.zeros_like(wl_axis)+1e-5
    
    df = pd.read_csv(filename, sep=None, engine='python')
    df.columns = [c.strip() for c in df.columns]
    f_n = interp1d(df['wl'], df['n'], kind='linear', fill_value="extrapolate")
    f_k = interp1d(df['wl'], df['k'], kind='linear', fill_value="extrapolate")
    n_arr = f_n(wl_axis)
    k_arr = f_k(wl_axis)
    k_arr[k_arr < 0] = 1e-9
    return n_arr, k_arr

def model_tmm_advanced(wl_arr, d_um, n_film, k_film, n_sub, k_sub, theta_deg=0):
    """支持角度依赖的 TMM 模型"""
    theta_rad = np.deg2rad(theta_deg)
    N_air = 1.0
    N_film = n_film - 1j * k_film
    N_sub = n_sub - 1j * k_sub
    
    sin_t_film = (N_air / N_film) * np.sin(theta_rad)
    cos_t_film = np.sqrt(1 - sin_t_film**2)
    sin_t_sub = (N_air / N_sub) * np.sin(theta_rad)
    cos_t_sub = np.sqrt(1 - sin_t_sub**2)
    
    delta = 2 * np.pi * N_film * d_um * cos_t_film / wl_arr
    
    # 导纳
    eta_film_TE = N_film * cos_t_film; eta_sub_TE = N_sub * cos_t_sub; eta_air_TE = N_air * np.cos(theta_rad)
    eta_film_TM = N_film / cos_t_film; eta_sub_TM = N_sub / cos_t_sub; eta_air_TM = N_air / np.cos(theta_rad)
    
    cos_d = np.cos(delta); sin_d = np.sin(delta)
    
    def solve_R(eta_0, eta_f, eta_s):
        M11 = cos_d
        M12 = (1j / eta_f) * sin_d
        M21 = 1j * eta_f * sin_d
        M22 = cos_d
        B = M11 + M12 * eta_s
        C = M21 + M22 * eta_s
        r = (eta_0 * B - C) / (eta_0 * B + C)
        return np.abs(r)**2

    R_TE = solve_R(eta_air_TE, eta_film_TE, eta_sub_TE)
    R_TM = solve_R(eta_air_TM, eta_film_TM, eta_sub_TM)
    return 1 - 0.5 * (R_TE + R_TM)

def calc_hemi_emissivity(wl_arr, d_um, n_p, k_p, n_s, k_s):
    """计算半球发射率"""
    angles = np.linspace(0, 89.5, 15) # 简化采样以加快速度
    rads = np.deg2rad(angles)
    E_ang_matrix = []
    for ang in angles:
        E_ang_matrix.append(model_tmm_advanced(wl_arr, d_um, n_p, k_p, n_s, k_s, ang))
    E_ang_matrix = np.array(E_ang_matrix)
    weights = 2 * np.sin(rads) * np.cos(rads)
    E_hemi = np.trapz(E_ang_matrix * weights[:, None], rads, axis=0)
    return E_hemi

def planck_law(wl_um, T):
    c2 = 14388
    return 1.0 / (wl_um**5 * (np.exp(c2 / (wl_um * T)) - 1))

# ==============================================================================
# 2. 环境与计算
# ==============================================================================
# 准备材料数据
n_pdms, k_pdms = load_material_from_csv('pdms_data.csv', 'PDMS')
n_ag, k_ag = load_material_from_csv('ag_data.csv', 'Ag')

# 环境辐射定义
def get_atmosphere_emissivity(wl_arr):
    e_atm = np.ones_like(wl_arr)
    for i, lam in enumerate(wl_arr):
        if 8.0 <= lam <= 13.0:
            e_atm[i] = 0.2 + 0.1 * (abs(lam - 10.5) / 2.5)**2
        else:
            e_atm[i] = 1.0
    return e_atm

def get_solar_spectrum(wl_arr):
    I_sun_bb = planck_law(wl_arr, 5800)
    total_power = np.trapz(I_sun_bb, wl_arr)
    return I_sun_bb * (1000.0 / total_power)

E_atm = get_atmosphere_emissivity(wl_axis)
I_sun = get_solar_spectrum(wl_axis)

def calculate_cooling_power(d_um, T_dev, T_amb=298.15, h_c=8.0):
    hemi_E = calc_hemi_emissivity(wl_axis, d_um, n_pdms, k_pdms, n_ag, k_ag)
    
    I_bb_dev = planck_law(wl_axis, T_dev)
    P_rad = np.trapz(hemi_E * I_bb_dev, wl_axis)
    
    I_bb_amb = planck_law(wl_axis, T_amb)
    P_atm = np.trapz(hemi_E * E_atm * I_bb_amb, wl_axis)
    
    P_sun = np.trapz(hemi_E * I_sun, wl_axis)
    P_cond = h_c * (T_amb - T_dev)
    
    P_net_day = P_rad - P_atm - P_sun - P_cond
    P_net_night = P_rad - P_atm - P_cond
    return P_net_day, P_net_night

# ==============================================================================
# 3. 主执行与绘图
# ==============================================================================
if __name__ == "__main__":
    print(">>> 正在运行 Q2: 辐射制冷性能评估...")
    
    # 1. 扫描厚度
    thickness_list = np.linspace(10, 150, 15)
    pow_day = []
    pow_night = []
    
    print("计算不同厚度的制冷功率...")
    for d in thickness_list:
        pd, pn = calculate_cooling_power(d, 298.15)
        pow_day.append(pd)
        pow_night.append(pn)
        
    # 绘图：功率 vs 厚度
    plt.figure(figsize=(10, 6))
    plt.plot(thickness_list, pow_night, 'o--', label='Nighttime', color='navy')
    plt.plot(thickness_list, pow_day, 's-', label='Daytime', color='orange')
    plt.title('Net Cooling Power vs. Thickness', fontsize=14)
    plt.xlabel('PDMS Thickness ($\\mu m$)')
    plt.ylabel('Net Cooling Power ($W/m^2$)')
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.show()
    
    # 2. 性能曲线 (特定厚度)
    d_opt = 75
    delta_T = np.linspace(0, 20, 30)
    p_curve = []
    for dt in delta_T:
        pd, _ = calculate_cooling_power(d_opt, 298.15 - dt)
        p_curve.append(pd)
        
    plt.figure(figsize=(10, 6))
    plt.plot(delta_T, p_curve, 'D-', color='teal', linewidth=2)
    plt.title(f'Cooling Performance Curve (d={d_opt} $\\mu m$)', fontsize=14)
    plt.xlabel('Temperature Drop (K)')
    plt.ylabel('Cooling Power ($W/m^2$)')
    plt.axhline(0, color='black'); plt.grid(True)
    plt.show()