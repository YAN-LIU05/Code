# src/main.py
"""
项目主执行文件。
"""
import matplotlib.pyplot as plt
import warnings
import config as cfg
from data_processing import load_cleaned_data
# ** 导入我们新的、职责分离的两个函数 **
from analysis_and_modeling import perform_base_analysis, find_and_show_best_interaction_model


def main():
    warnings.filterwarnings('ignore', category=FutureWarning)

    print("========== 开始执行建模分析流程 ==========")

    # 1. 加载数据
    all_needed_columns = list(set(
        [cfg.TARGET_VARIABLE] + cfg.NUMERICAL_FEATURES
    ))
    df = load_cleaned_data(
        filepath=cfg.CLEANED_DATA_PATH,
        columns_to_load=all_needed_columns
    )

    if df is not None and not df.empty:
        # 2. 执行基准分析，并获取返回的核心特征和R²
        base_features, baseline_adj_r2 = perform_base_analysis(
            df=df,
            target_var=cfg.TARGET_VARIABLE,
            top_n_features=cfg.TOP_N_FEATURES
        )

        # 3. 基于基准分析的结果，去寻找并展示增强模型
        if base_features:
            find_and_show_best_interaction_model(
                df=df,
                target_var=cfg.TARGET_VARIABLE,
                base_features=base_features,
                interaction_candidates=cfg.INTERACTION_CANDIDATES,
                baseline_adj_r2=baseline_adj_r2  # 将基准R²传入
            )

    print("\n========== 所有流程执行完毕 ==========")


if __name__ == "__main__":
    main()