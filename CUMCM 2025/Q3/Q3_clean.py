import pandas as pd
import numpy as np
import re


def clean_and_process_pregnancy_data(input_path, output_path):
    """
    对孕妇NIPT（无创产前检测）数据进行清洗、转换、聚合和特征工程。

    Args:
        input_path (str): 输入数据文件的路径 (支持 .xlsx, .csv, .tsv)。
        output_path (str): 清洗后数据保存的路径 (推荐 .csv 或 .xlsx)。
    """
    # --- 1. 数据加载 ---
    print(f"开始从 '{input_path}' 加载数据...")
    try:
        if input_path.endswith('.xlsx'):
            # 使用 openpyxl 引擎读取 Excel 文件
            df = pd.read_excel(input_path, engine='openpyxl')
        elif input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.tsv'):
            df = pd.read_csv(input_path, sep='\t')
        else:
            raise ValueError("不支持的文件格式，请提供 .xlsx, .csv 或 .tsv 文件。")
        print(f"成功加载 {len(df)} 条记录。")
    except FileNotFoundError:
        print(f"错误：文件未找到于 '{input_path}'")
        return
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    # --- 2. 数据清洗与格式统一 ---
    print("开始数据清洗和转换...")

    # 2.1 统一日期格式
    # 对于Excel，日期可能已经是datetime类型，但为确保统一，再次转换
    df['末次月经'] = pd.to_datetime(df['末次月经'], errors='coerce')
    # 检测日期格式为 20230429，需要指定 format
    df['检测日期'] = pd.to_datetime(df['检测日期'], format='%Y%m%d', errors='coerce')

    # 2.2 转换分类数据为数值
    df['IVF妊娠_数值'] = df['IVF妊娠'].map({'自然受孕': 0, 'IVF（试管婴儿）': 0, 'IUI（人工授精）': 1})
    df['胎儿是否健康_数值'] = df['胎儿是否健康'].map({'是': 1, '否': 0})

    # 2.3 处理 '染色体的非整倍体' -> 检测结果是否阳性
    # 只要字符串中包含 'T'，就认为是阳性(1)，否则为阴性(0)。fillna(False)处理空值情况。
    df['检测结果_阳性'] = df['染色体的非整倍体'].astype(str).str.contains('T', na=False).astype(int)

    # 2.4 处理 '怀孕次数'
    # 将 '≥3' 替换为 3，然后转换为数值
    df['怀孕次数'] = df['怀孕次数'].astype(str).replace('≥3', '3')
    df['怀孕次数_数值'] = pd.to_numeric(df['怀孕次数'], errors='coerce')

    # 2.5 统一 '检测孕周'
    # 使用正则表达式提取 'w' 前的数字

    df['检测孕周_周数'] = df['检测孕周'].astype(str).str.extract(r'(\d+)').astype(float)
    # 提取 'w' 后的数字
    # 提取 'w' 后的数
    # 提取 'w' 后面的数字
    df['检测孕周_天数'] = df['检测孕周'].astype(str).str.extract(r'w\+?(\d+)').astype(float)
    df['检测孕期时间'] = df['检测孕周_周数'] + df['检测孕周_天数']/7.0
    print("数据清洗和转换完成。")

    # --- 3. 处理同一孕妇单日多次抽血的情况 ---
    print("开始处理重复抽血记录...")

    # 识别重复记录
    duplicates = df.duplicated(subset=['孕妇代码', '检测日期'], keep=False)
    if duplicates.any():
        print(f"发现 {duplicates.sum()} 条重复抽血记录，将进行聚合处理。")

        # 确定数值型列和非数值型列
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # 移除我们手动创建的、需要特殊处理的数值列
        numeric_cols.remove('检测结果_阳性')
        numeric_cols.remove('胎儿是否健康_数值')

        agg_rules = {col: 'mean' for col in numeric_cols}

        # 添加非数值列和特殊处理列的规则
        agg_rules.update({
            '末次月经': 'first', 'IVF妊娠': 'first', '怀孕次数': 'first',
            'IVF妊娠_数值': 'first', '怀孕次数_数值': 'first', '胎儿是否健康': 'first',

            # 逻辑判断列：只要有一次阳性/不健康，结果就为阳性/不健康
            '检测结果_阳性': 'max',  # 0和1中取max，相当于OR逻辑
            '胎儿是否健康_数值': 'min',  # 1(健康)和0(不健康)中取min，只要有0就为0
        })

        # 确保聚合字典中的所有键都存在于DataFrame中
        agg_rules_final = {k: v for k, v in agg_rules.items() if k in df.columns}

        df_agg = df.groupby(['孕妇代码', '检测日期']).agg(agg_rules_final).reset_index()

        df_no_duplicates = df[~duplicates]
        df_processed = pd.concat([df_no_duplicates, df_agg], ignore_index=True)

    else:
        print("未发现重复抽血记录。")
        df_processed = df

    # --- 4. 新增判断列 (特征工程) ---
    print("新增检测准确性判断列...")

    conditions = [
        (df_processed['检测结果_阳性'] == 1) & (df_processed['胎儿是否健康_数值'] == 0),
        (df_processed['检测结果_阳性'] == 0) & (df_processed['胎儿是否健康_数值'] == 1),
        (df_processed['检测结果_阳性'] == 1) & (df_processed['胎儿是否健康_数值'] == 1),
        (df_processed['检测结果_阳性'] == 0) & (df_processed['胎儿是否健康_数值'] == 0)
    ]

    choices = [
        '1',
        '2',
        '3',
        '4'
    ]
   # '正确 (真阳性)',    '正确 (真阴性)',    '假阳性',    '假阴性'
    df_processed['检测准确性'] = np.select(conditions, choices, default='未知/数据不全')
    # --- 5. 新增判断列 是否达标 ---

    # --- 6. 整理并保存结果 ---
    print("开始整理并保存最终结果...")

    # 清理可能存在的列名中的空格
    df_processed.columns = df_processed.columns.str.strip()

    final_columns = [
        '孕妇代码', '检测日期', '年龄', '身高', '体重', '孕妇BMI', '末次月经','检测孕周_周数','检测孕周_天数', '检测孕期时间',
        'IVF妊娠', 'IVF妊娠_数值', '怀孕次数', '怀孕次数_数值', '生产次数',
        '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',
        '染色体的非整倍体',
        '检测结果_阳性', '胎儿是否健康', '胎儿是否健康_数值', '检测准确性',
        '原始读段数', '唯一比对的读段数', 'GC含量', 'Y染色体浓度', 'X染色体浓度','Y浓度是否达标'
    ]
    final_columns_exist = [col for col in final_columns if col in df_processed.columns]

    final_df = df_processed[final_columns_exist]

    try:
        if output_path.endswith('.csv'):
            # 使用 utf-8-sig 编码确保中文在Excel中正确显示
            final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif output_path.endswith('.xlsx'):
            final_df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"处理完成！结果已保存至 '{output_path}'")
    except Exception as e:
        print(f"保存文件时出错: {e}")


if __name__ == '__main__':
    # --- 配置输入和输出文件路径 ---
    # !! 请将这里替换为您的 Excel 输入文件名 !!
    INPUT_FILE = '../data.xlsx'

    # 您可以选择输出为 .csv (推荐) 或 .xlsx 文件
    OUTPUT_FILE = 'cleaned_data_origin.xlsx'

    # --- 执行主函数 ---
    clean_and_process_pregnancy_data(INPUT_FILE, OUTPUT_FILE)