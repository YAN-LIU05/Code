import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 假设df为你的原始DataFrame，且包含下列所有列
df = pd.read_excel('data.xlsx')

# 自动构造人工特征（如果未包含）
df['(FEV3-FEV1)/FVC'] = (df['FEV3'] - df['FEV1']) / df['FVC']

# 定义特征列（人工特征放最后）
feature_cols = [
    'Male', 'Age', 'Height', 'Weight', 'BMI',
    'WBC', 'EOS', 'FVC', 'FVC%', 'FEV1', 'FEV1%',
    'FEV1/FVC', 'PEF', 'PEF%', 'MMEF', 'MMEF%',
    'MEF75', 'MEF75%', 'MEF50', 'MEF50%',
    'MEF25', 'MEF25%', 'FEV3', '(FEV3-FEV1)/FVC'
]

X = df[feature_cols]
y = df['BHR']

# 标准化特征（增强模型稳定性）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 训练随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 获取特征重要性
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# 可视化
plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=np.array(feature_cols)[indices])
plt.title("随机森林特征重要性排序")
plt.xlabel("特征重要性得分")
plt.ylabel("特征")
plt.tight_layout()
plt.savefig("feature_importance.png")

# 输出前8个重要特征
top_8 = list(np.array(feature_cols)[indices[:8]])
print("前8个重要特征：", top_8)
