from sklearn.ensemble import RandomForestClassifier
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

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("随机森林特征重要性：\n", importances)
