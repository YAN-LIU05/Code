# src/analysis_and_modeling.py
"""
执行完整的分析和建模流程。
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import itertools

# 将字体设置放在文件顶部，全局生效
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def perform_base_analysis(df, target_var, top_n_features):
    """
    执行基础的相关性分析和OLS建模，并返回选定的核心特征。
    """
    # --- 1. 相关性分析与特征选择 ---
    print("\n" + "=" * 25 + " 1. 相关性分析与特征初选 " + "=" * 25)
    correlations = df.corr(method='pearson')[target_var].abs().sort_values(ascending=False)
    base_features = correlations[1:top_n_features + 1].index.tolist()
    print(f"根据相关性，初步选择 {len(base_features)} 个特征进入模型: {base_features}")

    # 绘制热力图
    plt.figure(figsize=(12, 10));
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5);
    plt.title('所有变量的相关性热力图 (Heatmap)');
    plt.show()

    # --- 2. 建立并展示基准OLS模型 ---
    print("\n" + "=" * 25 + " 2. 基准模型分析 (仅含主效应) " + "=" * 25)

    Y = df[target_var]
    X = df[base_features]

    # 标准化
    y_scaled = StandardScaler().fit_transform(Y.values.reshape(-1, 1)).flatten()
    X_scaled_df = pd.DataFrame(StandardScaler().fit_transform(X), columns=base_features, index=X.index)
    X_scaled_with_const = sm.add_constant(X_scaled_df)

    # 训练并展示模型
    base_model = sm.OLS(y_scaled, X_scaled_with_const).fit()
    print("\n--- 基准模型结果摘要 ---")
    print(base_model.summary())

    # --- 3. 可视化基准模型系数 ---
    params = base_model.params.drop('const')
    conf_int = base_model.conf_int().drop('const')
    errors = params - conf_int[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    params.plot(kind='bar', yerr=errors, ax=ax, capsize=5, color='skyblue', edgecolor='black')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title('基准模型中各特征的标准化系数')
    ax.set_ylabel('标准化系数 (Standardized Beta Coefficient)')
    ax.set_xlabel('特征')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 返回基准模型的关键信息，用于后续比较
    return base_features, base_model.rsquared_adj


def find_and_show_best_interaction_model(df, target_var, base_features, interaction_candidates, baseline_adj_r2):
    """
    系统性地测试交互项，并构建和展示包含最佳交互项的最终模型。
    """
    print("\n" + "=" * 25 + " 3. 特征工程：寻找并评估最佳交互项 " + "=" * 25)
    print(f"基准模型的调整后 R² 为: {baseline_adj_r2:.4f}，我们将以此为比较基准。")

    Y = df[target_var]
    y_scaled = StandardScaler().fit_transform(Y.values.reshape(-1, 1)).flatten()

    results = []
    # 遍历所有可能的交互项组合
    for f1, f2 in itertools.combinations(interaction_candidates, 2):
        df_temp = df.copy()
        interaction_term_name = f"{f1}_x_{f2}"
        df_temp[interaction_term_name] = df_temp[f1] * df_temp[f2]

        features_with_interaction = base_features + [interaction_term_name]
        X_new = df_temp[features_with_interaction]
        X_new_scaled_df = pd.DataFrame(StandardScaler().fit_transform(X_new), columns=features_with_interaction,
                                       index=X_new.index)

        new_model = sm.OLS(y_scaled, sm.add_constant(X_new_scaled_df)).fit()
        results.append({
            "Interaction": interaction_term_name,
            "New_Adj_R2": new_model.rsquared_adj,
            "Improvement": new_model.rsquared_adj - baseline_adj_r2,
            "PValue": new_model.pvalues[interaction_term_name]
        })

    if not results:
        print("没有生成任何可测试的交互项。")
        return

    results_df = pd.DataFrame(results).sort_values(by="Improvement", ascending=False)
    print("\n--- 所有交互项测试结果汇总 ---")
    print(results_df)

    best_candidate_df = results_df[(results_df['Improvement'] > 0) & (results_df['PValue'] < 0.05)]

    if best_candidate_df.empty:
        print("\n--- 结论 ---")
        print("没有找到任何能够显著改善模型的交互项。建议使用上方的基准模型。")
        return

    # --- 如果找到了最佳交互项，则构建并展示增强模型 ---
    best_interaction_name = best_candidate_df.iloc[0]['Interaction']
    print(f"\n已找到最佳交互项: '{best_interaction_name}'")

    print("\n" + "=" * 25 + " 4. 增强模型分析 (含最佳交互项) " + "=" * 25)

    df[best_interaction_name] = df[best_interaction_name.split('_x_')[0]] * df[best_interaction_name.split('_x_')[1]]
    final_features = base_features + [best_interaction_name]

    X_final = df[final_features]
    X_final_scaled_df = pd.DataFrame(StandardScaler().fit_transform(X_final), columns=final_features,
                                     index=X_final.index)
    X_final_with_const = sm.add_constant(X_final_scaled_df)

    final_model = sm.OLS(y_scaled, X_final_with_const).fit()
    print("\n--- 增强模型结果摘要 ---")
    print(final_model.summary())

    # 可视化增强模型的系数
    params = final_model.params.drop('const')
    conf_int = final_model.conf_int().drop('const')
    errors = params - conf_int[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    params.plot(kind='bar', yerr=errors, ax=ax, capsize=5, color='darkcyan', edgecolor='black')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title('增强模型中各特征的标准化系数')
    ax.set_ylabel('标准化系数 (Standardized Beta Coefficient)')
    ax.set_xlabel('特征')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()