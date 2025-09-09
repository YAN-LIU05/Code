import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 全局设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 增加备用字体以提高兼容性
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 加载数据 ---
source_file = 'data.xlsx'
cleaned_file = 'data_cleaned2.xlsx'

try:
    df_male_raw = pd.read_excel(source_file, sheet_name='男胎检测数据')
    df_female_raw = pd.read_excel(source_file, sheet_name='女胎检测数据')
    print(f"成功从 '{source_file}' 加载数据。")
except Exception as e:
    print(
        f"错误: 加载Excel文件失败。请检查文件 '{source_file}' 是否存在，并且包含 '男胎检测数据' 和 '女胎检测数据' sheets。")
    print(f"具体错误: {e}")
    df_male_raw = pd.DataFrame()
    df_female_raw = pd.DataFrame()


# --- 3. 数据清洗与筛选函数 (已更新) ---
def clean_nipt_data(df, is_male_sheet=True):
    """对NIPT数据进行全面的、更健壮的清洗和质控筛选"""
    if df.empty:
        return pd.DataFrame()

    df_cleaned = df.copy()
    initial_rows = len(df_cleaned)
    print(f"\n--- 开始清洗 {'男胎' if is_male_sheet else '女胎'} 数据 (原始 {initial_rows} 行) ---")

    # 打印列名用于调试
    # print("读取到的列名列表:", df_cleaned.columns.tolist())

    # 3.1 类型转换与格式统一
    if '检测孕周' in df_cleaned.columns:
        df_cleaned['孕周_数值'] = df_cleaned['检测孕周'].astype(str).str.extract(r'(\d+)w\+(\d)').apply(
            lambda x: pd.to_numeric(x[0], 'coerce') + pd.to_numeric(x[1], 'coerce') / 7, axis=1
        )

    numeric_cols = [
        '年龄', '身高', '体重', '孕妇BMI', 'Y染色体浓度', 'Y染色体的Z值',
        '怀孕次数', '生产次数', 'GC含量', '被过滤掉读段数的比例',
        '在参考基因组上比对的比例', '唯一比对的读段数'
    ]
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        else:
            print(f"警告: 列 '{col}' 不存在，跳过数值转换。")

    # 3.2 缺失值处理
    if all(c in df_cleaned.columns for c in ['孕妇BMI', '身高', '体重']):
        bmi_missing_mask = df_cleaned['孕妇BMI'].isnull() & df_cleaned['身高'].notnull() & df_cleaned['体重'].notnull()
        df_cleaned.loc[bmi_missing_mask, '孕妇BMI'] = df_cleaned['体重'] / ((df_cleaned['身高'] / 100) ** 2)

    core_vars = ['孕周_数值', '孕妇BMI']
    if is_male_sheet: core_vars.append('Y染色体浓度')
    df_cleaned.dropna(subset=[v for v in core_vars if v in df_cleaned.columns], inplace=True)
    print(f"删除核心变量缺失行后，剩余 {len(df_cleaned)} 行。")

    # 3.3 基于生理和妊娠常识的异常值过滤
    if '年龄' in df_cleaned.columns: df_cleaned = df_cleaned[df_cleaned['年龄'].between(18, 55)]
    if '孕妇BMI' in df_cleaned.columns: df_cleaned = df_cleaned[df_cleaned['孕妇BMI'].between(15, 50)]
    if '孕周_数值' in df_cleaned.columns: df_cleaned = df_cleaned[df_cleaned['孕周_数值'].between(10, 42)]
    print(f"根据年龄/BMI/孕周常规范围过滤后，剩余 {len(df_cleaned)} 行。")

    # 3.4 严格数据质控 (QC) 筛选
    print("开始执行严格数据质控(QC)筛选...")
    qc_filters = {
        '在参考基因组上比对的比例': lambda df: df['在参考基因组上比对的比例'] > 0.7,
        '唯一比对的读段数': lambda df: df['唯一比对的读段数'] > 2000000,
        '被过滤掉读段数的比例': lambda df: df['被过滤掉读段数的比例'] < 0.05,
        'GC含量': lambda df: df['GC含量'].between(0.395, 0.6)
    }
    if is_male_sheet:
        qc_filters['Y染色体浓度'] = lambda df: df['Y染色体浓度'] > 0.04

    for col, condition in qc_filters.items():
        if col in df_cleaned.columns:
            rows_before = len(df_cleaned)
            df_cleaned = df_cleaned[condition(df_cleaned)]
            rows_after = len(df_cleaned)
            if rows_before > rows_after:
                print(f"  - 筛选 [{col}] 条件后，剩余 {rows_after} 行 (移除了 {rows_before - rows_after} 行)。")
        else:
            print(f"  - 警告: 质控列 '{col}' 不存在，跳过此项筛选。")

    # 3.5 特定Sheet的清洗规则
    if not is_male_sheet:
        if 'Y染色体浓度' in df_cleaned.columns and 'Y染色体的Z值' in df_cleaned.columns:
            female_y_present_mask = df_cleaned['Y染色体浓度'].notna() | df_cleaned['Y染色体的Z值'].notna()
            if female_y_present_mask.any():
                df_cleaned = df_cleaned[~female_y_present_mask]
            print(f"  - 女胎数据: 检查并移除存在Y染色体信息的记录后，剩余 {len(df_cleaned)} 行。")

    final_rows = len(df_cleaned)
    print(f"清洗完成，最终剩余 {final_rows} 行 (总计移除 {initial_rows - final_rows} 行)。")
    return df_cleaned


# --- 4. 数据探索与可视化函数 (已更新) ---
def plot_data_insights(df, title_prefix):
    """对DataFrame进行探索性可视化，并将图表保存为唯一文件名"""
    if df.empty:
        print(f"{title_prefix} DataFrame为空，跳过可视化。")
        return

    print(f"\n--- 正在生成 {title_prefix} 的可视化图表 ---")
    df = df.copy()

    # 转换孕周用于绘图 (如果清洗函数已创建，这里会覆盖，不影响结果)
    if '检测孕周' in df.columns and '孕周_数值' not in df.columns:
        df['孕周_数值'] = df['检测孕周'].astype(str).str.extract(r'(\d+)w\+(\d)').apply(
            lambda x: pd.to_numeric(x[0], 'coerce') + pd.to_numeric(x[1], 'coerce') / 7, axis=1
        )

    key_cols = ['年龄', '孕妇BMI', '孕周_数值', 'Y染色体浓度', 'GC含量', '在参考基因组上比对的比例', '唯一比对的读段数']

    # 筛选出DataFrame中真实存在的列进行绘图
    cols_to_plot = [col for col in key_cols if col in df.columns and df[col].notna().any()]

    if not cols_to_plot:
        print(f"在 {title_prefix} 中没有找到可供可视化的关键变量。")
        return

    plt.figure(figsize=(15, 10))
    plt.suptitle(f'{title_prefix} - 关键变量分布', fontsize=16)
    for i, col in enumerate(cols_to_plot):
        plt.subplot(3, 3, i + 1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(col)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 【修改】创建唯一文件名并保存图表
    safe_filename = title_prefix.replace(' ', '_').replace('-', '').lower() + "_insights.png"
    plt.savefig(safe_filename)
    print(f"图表已保存为: '{safe_filename}'")
    plt.show()


# --- 5. 清洗前数据可视化 ---
plot_data_insights(df_male_raw, "男胎 - 清洗前")
plot_data_insights(df_female_raw, "女胎 - 清洗前")

# --- 6. 执行清洗 ---
df_male_cleaned = clean_nipt_data(df_male_raw, is_male_sheet=True)
df_female_cleaned = clean_nipt_data(df_female_raw, is_male_sheet=False)

# --- 7. 清洗后数据可视化 (新增步骤) ---
print("\n" + "=" * 20 + " 生成清洗后数据的可视化图表 " + "=" * 20)
plot_data_insights(df_male_cleaned, "男胎 - 清洗后")
plot_data_insights(df_female_cleaned, "女胎 - 清洗后")

# --- 8. 保存结果 ---
if not df_male_cleaned.empty or not df_female_cleaned.empty:
    try:
        with pd.ExcelWriter(cleaned_file) as writer:
            if not df_male_cleaned.empty:
                df_male_cleaned.to_excel(writer, sheet_name='男胎数据_已清洗', index=False)
            if not df_female_cleaned.empty:
                df_female_cleaned.to_excel(writer, sheet_name='女胎数据_已清洗', index=False)
        print(f"\n成功！已将清洗和筛选后的数据保存至新文件: '{cleaned_file}'")
    except Exception as e:
        print(f"\n错误: 保存文件失败。请检查是否有关闭 '{cleaned_file}' 文件。")
        print(f"具体错误: {e}")
else:
    print("\n清洗后没有剩余数据，未生成新的Excel文件。")