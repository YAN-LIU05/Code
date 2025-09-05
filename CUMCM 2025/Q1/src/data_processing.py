# src/data_processing.py
"""
数据加载与预处理模块。
"""
import pandas as pd


# ** 将函数定义恢复为接收 'columns_to_load' **
def load_cleaned_data(filepath, columns_to_load):
    """
    加载数据，仅保留指定的列，并进行数据清洗。

    参数:
        filepath (str): Excel文件路径。
        columns_to_load (list): 一个包含所有需要保留的列名的列表。
    """
    try:
        df = pd.read_excel(filepath, sheet_name='男胎数据_已清洗')
        print(f"成功从 '{filepath}' 加载数据。原始维度: {df.shape}")
    except FileNotFoundError:
        print(f"错误: 文件未找到 '{filepath}'。")
        return None
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return None

    # --- 1. 检查并筛选指定的列 ---
    missing_cols = [col for col in columns_to_load if col not in df.columns]
    if missing_cols:
        print(f"错误: 数据文件中缺少以下必需的列: {missing_cols}")
        return None

    df = df[columns_to_load].copy()
    print(f"已从原始数据中筛选出 {len(columns_to_load)} 列用于全部后续分析。")

    # --- 2. 处理特殊值，如 '≥3' ---
    for col in ['怀孕次数', '生产次数']:
        if col in df.columns:
            df[col] = df[col].replace('≥3', 3)

    # --- 3. 将所有分析列转换为数值型并处理空值 ---
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"数据类型转换和空值处理后，剩余 {len(df)} / {initial_rows} 条记录。")

    if df.empty:
        print("警告：数据清洗后，没有数据剩余！请检查原始数据。")
        return None

    return df