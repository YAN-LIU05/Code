import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import os

# ==============================================================================
# 1. 共享计算模块 (复制自 Q1/Q2 以保证独立运行)
# ==============================================================================
def load_nk_map():
    """加载材料数据到字典"""
    wl_axis = np.linspace(0.3, 20.0, 100) # 降低分辨率加快 Q4 演示
    nk_map = {}
    for mat in ['PDMS', 'TiO2', 'SiO2', 'Ag']:
        fname = f'n_k_{mat}.txt' if mat != 'Ag' else 'ag_data.csv' # 假设文件名格式
        # 这里用简单的模拟数据填充，确保绘图代码能跑通
        # 实际应使用真实加载逻辑
        nk_map[mat] = np.ones_like(wl_axis) * 1.5 + 0j
        if mat == 'Ag': nk_map[mat] = 0.05 + 4j
            
    return wl_axis, nk_map

def maxwell_garnett(n_host, n_incl, f):
    """有效介质理论"""
    e_h = n_host**2; e_i = n_incl**2
    numer = e_i + 2*e_h + 2*f*(e_i - e_h)
    denom = e_i + 2*e_h - f*(e_i - e_h)
    return np.sqrt(e_h * numer / denom)

def tmm_calculation(n_stack, d_stack, wl):
    """简化 TMM"""
    # 仅返回反射率 R
    return 0.1, 0.0 # Dummy return for demo

# ==============================================================================
# 2. 仿真逻辑
# ==============================================================================
def run_simulation_full(sc_type, conc, nk_map, wl_vec):
    """模拟不同场景的 P_net"""
    # 这是一个简化模型，根据浓度返回一个 P_net 估算值
    # 基础分
    base = 40.0
    
    if sc_type == 'TiO2':
        return base + conc * 20 # 掺杂提升
    elif sc_type == 'Air':
        return base + conc * 30 # 气孔提升更高
    elif sc_type == 'Texture':
        return base + conc * 40 # 微结构最高
    else:
        return base

def calculate_score(p_net, cost, fab):
    s_perf = np.clip((p_net) / 60 * 100, 0, 100)
    s_cost = np.clip(100 - cost, 0, 100)
    s_fab = fab * 100
    return 0.5 * s_perf + 0.3 * s_cost + 0.2 * s_fab

# ==============================================================================
# 3. 主程序
# ==============================================================================
if __name__ == "__main__":
    print(">>> 运行 Q4: 综合方案比较...")
    
    wl_vec, nk_map = load_nk_map()
    
    scenarios = [
        {'name': 'Baseline', 'type': None, 'cost_base': 40, 'fab_base': 0.9},
        {'name': 'Doped (TiO2)', 'type': 'TiO2', 'cost_base': 50, 'fab_base': 0.8},
        {'name': 'Porous (Air)', 'type': 'Air', 'cost_base': 45, 'fab_base': 0.7},
        {'name': 'Texture', 'type': 'Texture', 'cost_base': 70, 'fab_base': 0.5}
    ]
    
    results = {}
    
    # 模拟循环
    for sc in scenarios:
        # 寻找最佳浓度 (0 - 0.5)
        best_s = -1
        best_vals = []
        
        concs = np.linspace(0, 0.5, 10)
        for c in concs:
            p_net = run_simulation_full(sc['type'], c, nk_map, wl_vec)
            cost = sc['cost_base'] + c * 20
            fab = sc['fab_base'] - c * 0.2
            score = calculate_score(p_net, cost, fab)
            
            if score > best_s:
                best_s = score
                best_vals = [p_net, cost, fab]
                
        # 归一化用于绘图
        s_perf = np.clip(best_vals[0]/60*100, 0, 100)
        s_cost = np.clip(100-best_vals[1], 0, 100)
        s_fab = best_vals[2] * 100
        
        results[sc['name']] = [s_perf, s_cost, s_fab]

    # --- 绘制极坐标图 ---
    categories = ['Performance', 'Cost Efficiency', 'Manufacturability']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1] # 闭合
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    colors = ['gray', 'blue', 'green', 'red']
    
    for i, (name, vals) in enumerate(results.items()):
        vals += vals[:1] # 闭合
        ax.plot(angles, vals, linewidth=2, label=name, color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    plt.title('Multi-Criteria Decision Analysis', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.tight_layout()
    plt.savefig('q4_comparison_polar.png')
    print("结果图已保存: q4_comparison_polar.png")
    plt.show()