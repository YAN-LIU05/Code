# diagnose_missing_data.py
import pandas as pd

# --- 从你的主项目中复制这些配置 ---
CLEANED_DATA_PATH = '../data/data_cleaned2.xlsx'  # 注意路径可能需要调整
# 这个列表应该和你 config.py 中的保持一致
COLUMNS_TO_CHECK = [
    'Y染色体浓度', '年龄', '孕周_数值', '孕妇BMI', '怀孕次数', '生产次数'
    # ... 把你所有用到的列都加进来 ...
]


def find_missing_rows(filepath, columns):
    """
    一个专门用来诊断数据缺失的函数。
    """
    try:
        df = pd.read_excel(filepath, sheet_name='男胎数据_已清洗')
    except Exception as e:
        print(f"加载文件失败: {e}")
        return

    # 筛选我们关心的列
    df_check = df[columns].copy()

    # 清理和转换
    for col in ['怀孕次数', '生产次数']:
        if col in df_check.columns:
            df_check[col] = df_check[col].replace('≥3', 3)
    for col in df_check.columns:
        df_check[col] = pd.to_numeric(df_check[col], errors='coerce')

    # 找出含有空值的行
    rows_with_nan = df[df_check.isnull().any(axis=1)]

    if not rows_with_nan.empty:
        print("--- 在以下行中发现缺失值 ---")
        # 打印这些行的详细信息，让你能看到原始数据
        # 我们只显示我们关心的那几列
        print(rows_with_nan[columns])

        # 如果只想看索引
        # print(f"缺失值所在行索引: {rows_with_nan.index.tolist()}")
    else:
        print("在指定列中未发现缺失值。")


if __name__ == "__main__":
    find_missing_rows(CLEANED_DATA_PATH, COLUMNS_TO_CHECK)