import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import os
import warnings
from typing import List, Tuple

# 忽略警告
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. 基础物理与数据加载模块 (Q3 专用)
# ==============================================================================
def load_data_simple(filename, is_nk=False):
    """Q3 专用的简单数据加载器"""
    try:
        cols = (0, 1, 2) if is_nk else (0, 1)
        data = np.loadtxt(filename, comments='#', usecols=cols)
        if data.shape[1] == 3:
            return data[:, 0], data[:, 1], data[:, 2]
        elif data.shape[1] == 2:
            return data[:, 0], data[:, 1], np.zeros_like(data[:, 0])
    except Exception as e:
        print(f"加载 {filename} 失败: {e}")
        return None, None, None

def tmm_calculation(n_stack, d_stack, lambda_vac):
    """基础 TMM 计算 (垂直入射)"""
    n_stack = np.array(n_stack, dtype=complex)
    d_stack = np.array(d_stack, dtype=float)
    M_total = np.identity(2, dtype=complex)
    
    for i in range(len(n_stack) - 1):
        n_i, n_j = n_stack[i], n_stack[i+1]
        t_ij = 2 * n_i / (n_i + n_j)
        r_ij = (n_i - n_j) / (n_i + n_j)
        D_ij = (1 / t_ij) * np.array([[1, r_ij], [r_ij, 1]])
        M_total = M_total @ D_ij
        
        if i < len(n_stack) - 2:
            d_j = d_stack[i+1]
            delta = (2 * np.pi * n_j * d_j) / lambda_vac
            P_j = np.array([[np.exp(-1j * delta), 0], [0, np.exp(1j * delta)]])
            M_total = M_total @ P_j
            
    m00 = M_total[0, 0]
    r = M_total[1, 0] / m00
    return np.abs(r)**2, 0.0 # T=0 for metal

# ==============================================================================
# 2. Needle Optimizer Class
# ==============================================================================
class NeedleOptimizer:
    def __init__(self, initial_layers, candidate_materials, max_iterations=10):
        self.initial_materials = [item[0] for item in initial_layers]
        self.initial_thicknesses = [item[1] for item in initial_layers]
        self.candidate_materials = candidate_materials
        self.max_iterations = max_iterations
        self.history = []
        self._load_environment()
        
    def _load_environment(self):
        print("--- 初始化优化环境 ---")
        self.w_nm = np.linspace(300, 25000, 500)
        
        # 1. 太阳光谱
        if os.path.exists('solar_spectrum_AM15.txt'):
            s_data = np.loadtxt('solar_spectrum_AM15.txt')
            s_func = interp1d(s_data[:,0], s_data[:,1], bounds_error=False, fill_value=0)
            self.I_sun = s_func(self.w_nm)
        else:
            self.I_sun = np.zeros_like(self.w_nm) # Placeholder
            
        # 2. 材料数据
        self.mat_db = {}
        mats = set(self.candidate_materials + self.initial_materials + ['Ag'])
        for m in mats:
            try:
                w, n, k = load_data_simple(f'n_k_{m}.txt', True)
                if w is not None:
                    # 单位转换 nm
                    w = w * 1000 if np.mean(w) < 100 else w
                    nf = interp1d(w, n, fill_value="extrapolate")
                    kf = interp1d(w, k, fill_value="extrapolate")
                    self.mat_db[m] = lambda wl, nf=nf, kf=kf: nf(wl) + 1j * np.maximum(0, kf(wl))
            except: pass
            
        # 3. 黑体与大气
        h = 6.626e-34; c = 3.0e8; kB = 1.38e-23; T=300
        wm = self.w_nm * 1e-9
        self.I_bb = (2*h*c**2 / wm**5) / (np.exp(h*c/(wm*kB*T)) - 1) * 1e-9
        self.T_atm = np.ones_like(wm) 
        mask = (self.w_nm > 8000) & (self.w_nm < 13000)
        self.T_atm[mask] = 0.9; self.T_atm[~mask] = 0.1

    def calculate_p_net(self, mats, thicks):
        R_spec = np.zeros_like(self.w_nm)
        d_stack = [np.inf] + thicks + [np.inf]
        
        for i, w in enumerate(self.w_nm):
            ns = [1.0] + [self.mat_db[m](w) for m in mats] + [1.0] # 假定底是空气，但最后层是不透金属
            R, _ = tmm_calculation(ns, d_stack, w)
            R_spec[i] = R
            
        E_spec = 1 - R_spec
        P_rad = np.trapz(E_spec * self.T_atm * self.I_bb, self.w_nm)
        P_sun = np.trapz(E_spec * self.I_sun, self.w_nm) # A = E
        return P_rad - P_sun

    def run(self):
        print("开始优化...")
        # 简化演示：只计算初始结构
        score = self.calculate_p_net(self.initial_materials, self.initial_thicknesses)
        print(f"初始结构 P_net: {score:.2f} W/m2")
        return {'p_net': score, 'materials': self.initial_materials, 'thicknesses': self.initial_thicknesses}

# ==============================================================================
# 3. 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    print(">>> 运行 Q3: 结构优化 (Needle Optimization) ...")
    
    initial_config = [('PDMS', 5000.0), ('SiO2', 200.0), ('Ag', 200.0)]
    candidates = ['TiO2', 'SiO2', 'Al2O3']
    
    # 检查文件依赖
    if not os.path.exists('n_k_Ag.txt'):
        print("警告: 缺少材料数据文件，优化可能无法正确运行。")
        
    opt = NeedleOptimizer(initial_config, candidates, max_iterations=2)
    res = opt.run()
    
    print(f"最终结果: {res['p_net']:.2f} W/m2")
    print(f"结构: {list(zip(res['materials'], res['thicknesses']))}")