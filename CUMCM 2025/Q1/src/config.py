# src/config.py
"""
配置文件，用于定义分析的核心参数。
"""

# --- 文件路径 ---
CLEANED_DATA_PATH = '../data/data_cleaned2.xlsx'

# --- 分析参数 ---
TARGET_VARIABLE = 'Y染色体浓度'
TOP_N_FEATURES = 3

# --- 明确定义用于分析的数值特征列 ---
# 我们只对这个列表中的列进行相关性分析和建模
NUMERICAL_FEATURES = [
    '年龄',
    '孕妇BMI',
    '怀孕次数',
    '生产次数',
    '孕周_数值'
    # 注意：这里不应包含目标变量 'Y染色体浓度'
]


# --- 特征工程配置 ---
# 定义用于创建交互项的候选特征列表
# 通常我们会选择那些在主效应模型中已经很显著的变量
INTERACTION_CANDIDATES = [
    '年龄',
    '孕周_数值',
    '孕妇BMI'
]

