import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# --- 假设 'df' 是您预处理好的 DataFrame ---
# --- 并且您已经运行了之前的代码加载了 df ---
try:
    if 'df' not in locals():
        # 尝试加载，替换为您实际的文件和路径
        file_path = r"F:\A_MathModel\model\model\01data.xls" # 使用原始字符串
        df = pd.read_excel(file_path)
        print("成功加载数据。")
    else:
        print("使用内存中的 df 数据。")
except Exception as e:
    print(f"加载数据时出错: {e}")
    exit()

# --- 1. 准备用于 PCA 的数据 ---
# 选择要进行 PCA 的特征列。通常是所有的数值型预测变量。
# 需要排除目标变量 'BHR' 和可能不想包含的分类变量 'Male'（除非已独热编码）
# 这里假设我们使用之前定义的 numerical_cols (需要确保它只包含你想做PCA的特征)
# 或者，手动定义/选择你想用于PCA的列名列表
features_for_pca = df.select_dtypes(include=np.number).columns.drop(['BHR'], errors='ignore')
# 如果 Male 是 0/1 编码且你想包含它，就不要从 features_for_pca 中移除
# 如果不想包含 Male, 确保移除: features_for_pca = features_for_pca.drop(['Male'], errors='ignore')

print(f"\n用于 PCA 的特征 ({len(features_for_pca)}):")
print(features_for_pca.tolist())

X = df[features_for_pca]

# --- 2. 数据标准化 (非常重要！) ---
# PCA 对数据的尺度非常敏感，必须先进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. 执行 PCA ---
# 您可以选择要保留的主成分数量 (n_components)
# 如果设置为 None，则会计算所有可能的主成分
# 为了得到所有成分与原始特征的关系，先不设置或设置为原始特征数
pca = PCA(n_components=None) # 或者 pca = PCA()

# 拟合 PCA 模型
pca.fit(X_scaled)

# --- 4. 获取成分/载荷矩阵 ---
# 这就是您需要的结果！
loadings = pca.components_

# --- 5. (推荐) 将结果整理成更易读的 DataFrame ---
# 创建一个 DataFrame 来显示载荷
# 行是主成分 (PC1, PC2, ...)
# 列是原始特征名 (按 X 中的顺序)
loadings_df = pd.DataFrame(loadings,
                           columns=features_for_pca, # 使用原始特征名作为列名
                           index=[f'PC{i+1}' for i in range(loadings.shape[0])]) # 创建 PC1, PC2... 作为行索引

print("\n--- PCA 成分/载荷矩阵 (Loadings Matrix) ---")
# 为了更好的显示效果，可以转置 DataFrame (可选)
import pandas as pd



# --- 在打印 loadings_df 之前设置这些选项 ---

# a) 显示所有列
pd.set_option('display.max_columns', None)

# b) 显示所有行 (如果行数也很多)
pd.set_option('display.max_rows', None)

# c) 设置每行显示的最大宽度 (字符数)，防止列换行
#    设置为一个较大的值，或者 None 尝试不限制（但可能受终端宽度影响）
#    可以尝试 1000 或更大
pd.set_option('display.width', 1000) # 或者 pd.set_option('display.width', None)

# d) （可选）设置单个单元格内容的最大宽度 (对数值矩阵影响不大)
# pd.set_option('display.max_colwidth', None)

# --- 现在打印你的 DataFrame ---
# 假设 loadings_df 是你的 PCA 载荷矩阵 DataFrame
try:
    if 'loadings_df' in locals():
        print("\n--- PCA 成分/载荷矩阵 (完整显示) ---")
        print(loadings_df.round(3)) # 打印设置好选项后的 DataFrame
    else:
        print("错误：变量 'loadings_df' 不存在。")
except Exception as e:
    print(f"打印 DataFrame 时出错: {e}")

# --- （可选）在脚本末尾恢复默认设置，以免影响后续代码 ---
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')
# pd.reset_option('display.width')
# pd.reset_option('display.max_colwidth')


print(loadings_df.round(3)) # 显示小数点后3位

# 如果想看哪个特征对 PC1 贡献最大，可以查看第一行绝对值大的值
print("\n--- 对 PC1 贡献最大的特征 (按载荷绝对值排序) ---")
pc1_loadings = loadings_df.loc['PC1'].abs().sort_values(ascending=False)
print(pc1_loadings.head(10)) # 显示前10个

print("\n--- 对 PC2 贡献最大的特征 (按载荷绝对值排序) ---")
pc2_loadings = loadings_df.loc['PC2'].abs().sort_values(ascending=False)
print(pc2_loadings.head(10)) # 显示前10个

# --- 同时，您可以获取解释方差比（验证是否与您之前的结果一致） ---
explained_variance_ratio = pca.explained_variance_ratio_
print("\n--- 各主成分的解释方差比 ---")
print(explained_variance_ratio)