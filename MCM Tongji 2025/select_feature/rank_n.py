import pandas as pd
import matplotlib.pyplot as plt

# ====== ① 输入各方法的排名结果（按重要性从高到低） ======
lasso_rank = ['FEV1\FVC%', 'EOS', 'MEF50%', 'Age', 'FEV1%', 'Weight', 'MEF75', 'Male']
rf_rank = [
    'EOS',
    'MEF50%',
    'FEV1/FVC',
    '(FEV3-FEV1)/FVC',
    'MEF25%',
    'MMEF%',
    'MEF75%',
    'PEF%',
    'WBC',
    'BMI',
    'FEV1%',
    'MMEF',
    'MEF50',
    'FVC%',
    'MEF75',
    'Age',
    'PEF',
    'FEV1',
    'MEF25',
    'FVC',
    'Height',
    'FEV3',
    'Weight',
    'Male'
]
lgb_rank = [
    'EOS',
    'BMI',
    '(FEV3-FEV1)/FVC',
    'WBC',
    'PEF%',
    'MEF50%',
    'MEF25%',
    'FEV1/FVC',
    'FEV1%',
    'Age',
    'PEF',
    'MEF75%',
    'FVC%',
    'MEF25',
    'MMEF%',
    'Weight',
    'MEF75',
    'MEF50',
    'Height',
    'MMEF',
    'FVC',
    'FEV1',
    'FEV3',
    'Male'
]
pca_rank = [
    'FEV1/FVC',
    'MEF75%',
    'FEV1',
    'MEF50',
    'MEF25',
    'PEF%',
    'FEV3',
    'MEF50%',
    'Male',
    'MMEF%',
    'Height',
    'Weight',
    '(FEV3-FEV1)/FVC',
    'MMEF',
    'FVC',
    'MEF75',
    'Age',
    'FEV1%',
    'PEF',
    'MEF25%',
    'FVC%',
    'WBC',
    'EOS',
    'BMI'
]

# ====== ② 整合所有出现的特征名 ======
all_features = set(lasso_rank + rf_rank + lgb_rank + pca_rank)
#rank_df = pd.DataFrame(index=all_features)
rank_df = pd.DataFrame(index=list(all_features))  # ✅ 转换为 list

# ====== ③ 为每种方法计算排名，未出现的特征排名为99 ======
methods = {
    'Lasso': lasso_rank,
    'RandomForest': rf_rank,
    'LightGBM': lgb_rank,
    'PCA': pca_rank
}

for method_name, method_ranks in methods.items():
    rank_df[method_name] = rank_df.index.map(lambda f: method_ranks.index(f) + 1 if f in method_ranks else 99)

# ====== ④ 计算平均排名并排序 ======
rank_df['AverageRank'] = rank_df.mean(axis=1)
rank_df = rank_df.sort_values('AverageRank')

# ====== ⑤ 输出前10个特征排名 ======
print("综合排名前10的重要特征：\n")
print(rank_df.head(10))

# ====== ⑥ 可视化前10重要特征 ======
plt.figure(figsize=(10, 6))
rank_df['AverageRank'].head(10).sort_values().plot(kind='barh', color='skyblue')
plt.title("Ranking of the importance of comprehensive features(Top10)")
plt.xlabel("Average rank (smaller is more important)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_ranking_top10.png", dpi=300)
plt.show()
