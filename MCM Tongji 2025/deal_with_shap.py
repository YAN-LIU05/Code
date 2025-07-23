import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- Step 1：载入数据 ----------------
df = pd.read_excel('data.xlsx')

# 构造人工特征
df['(FEV3-FEV1)/FVC'] = (df['FEV3'] - df['FEV1']) / df['FVC']

# 特征列
feature_cols = [
    'Male', 'Age', 'Height', 'Weight', 'BMI',
    'WBC', 'EOS', 'FVC', 'FVC%', 'FEV1', 'FEV1%',
    'FEV1/FVC', 'PEF', 'PEF%', 'MMEF', 'MMEF%',
    'MEF75', 'MEF75%', 'MEF50', 'MEF50%',
    'MEF25', 'MEF25%', 'FEV3', '(FEV3-FEV1)/FVC'
]

# 检查是否有缺失值
if df[feature_cols].isnull().any().any():
    print("存在缺失值，请先处理！")
    exit()

# ---------------- Step 2：准备数据 ----------------
X = df[feature_cols]
y = df['BHR'].astype(int)  # 确保目标变量为整数0/1

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.3, random_state=42)

# ---------------- Step 3：训练模型 ----------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ---------------- Step 4：SHAP分析 ----------------
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# 检查 shap_values 类型
if isinstance(shap_values, list):
    # 多分类或二分类，使用类别1的SHAP值
    shap_summary = shap_values[1]
else:
    # 二分类时直接是一个 (n_samples, n_features) 数组
    shap_summary = shap_values


# ---------------- Step 5：SHAP 全局特征重要性图 ----------------

# ✅ 条形图：展示平均绝对贡献值
shap.summary_plot(shap_summary, X_test, feature_names=feature_cols, plot_type="bar")

# ✅ 蜂群图（可选）：展示每个样本的贡献分布
shap.summary_plot(shap_summary, X_test, feature_names=feature_cols)

plt.show()
