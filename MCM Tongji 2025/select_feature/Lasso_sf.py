from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 加载你清洗后的数据文件
df_clean = pd.read_excel("01data.xls")

# 设定你要选择的重要特征（根据任务要求）
feature_cols = ['Male',
'Age',
'Height',
'Weight',
'BMI',
'WBC',
'EOS',
'FVC',
'FVC%',
'FEV1',
'FEV1%',
'FEV1/FVC',
'PEF',
'PEF%',
'MMEF',
'MMEF%',
'MEF75',
'MEF75%',
'MEF50',
'MEF50%',
'MEF25',
'MEF25%',
'FEV3',
'(FEV3-FEV1)/FVC','MEF75%-MEF25%','MMEF/PEF','FEV3 - FEV1','(MEF50 + MEF25 + MEF75) / 3','EOS/WBC'
]
X = df_clean[feature_cols]
y = df_clean['BHR']

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(cv=5, random_state=0).fit(X_scaled, y)
lasso_coef = pd.Series(lasso.coef_, index=feature_cols)

# 筛选非零权重特征
important_lasso = lasso_coef[lasso_coef != 0].sort_values(ascending=False)
print("Lasso筛选的重要特征：\n", important_lasso)
