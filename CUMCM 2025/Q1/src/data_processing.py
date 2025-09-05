# src/data_processing.py
"""
数据加载与预处理模块。
"""
import pandas as pd


def load_cleaned_data(filepath, columns_to_load):
    """
    加载数据，筛选列，清洗，并将'怀孕次数'和'生产次数'中的空值(NaN)填充为3。
    """
    try:
        df = pd.read_excel(filepath, sheet_name='男胎数据_已清洗')
        print(f"成功从 '{filepath}' 加载数据。原始维度: {df.shape}")
    except Exception as e:
        print(f"加载文件失败: {e}")
        return None

    # --- 1. 筛选指定的列 ---
    df = df[columns_to_load].copy()

    # --- 2. 预处理计数类列 ---
    # 首先，像之前一样，将所有列尝试转换为数值，无法转换的会变成 NaN
    for col in df.columns:
        # 我们在这里先不替换'≥3'，让to_numeric直接处理它
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 3. 核心改动：将指定列的 NaN 填充为 3 ---
    cols_to_fill = ['怀孕次数', '生产次数']
    for col in cols_to_fill:
        if col in df.columns:
            # 记录填充前有多少个NaN
            nan_count_before = df[col].isnull().sum()
            if nan_count_before > 0:
                print(f"在列 '{col}' 中发现 {nan_count_before} 个空值，将它们填充为 3。")
                # 使用 .fillna() 进行填充
                df[col].fillna(3, inplace=True)
            else:
                print(f"列 '{col}' 中没有发现空值。")

    # --- 4. 识别并报告剩余的空值 (来自其他列) ---
    # 现在df中'怀孕次数'和'生产次数'应该没有NaN了
    rows_with_nan = df[df.isnull().any(axis=1)]

    if not rows_with_nan.empty:
        print("\n" + "=" * 20 + " 警告：在其他列中发现并即将删除以下含有空值的行 " + "=" * 20)
        print(f"受影响的行索引: {rows_with_nan.index.tolist()}")
        # 打印详细信息，帮助定位是哪个其他列出了问题
        for index, row in rows_with_nan.iterrows():
            nan_cols = row[row.isnull()].index.tolist()
            print(f"  - 行索引 {index}: 在列 {nan_cols} 中存在空值")
        print("=" * 70)
    else:
        print("\n数据质量良好，在所有选定列中均未发现空值。")

    # --- 5. 删除剩余的空值行 ---
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"最终数据清洗后，剩余 {len(df)} / {initial_rows} 条记录。")

    if df.empty:
        print("警告：数据清洗后，没有数据剩余！")
        return None

    return df