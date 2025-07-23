from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 sklearn 风格的 LightGBM 接口（推荐）
model = LGBMClassifier(
    objective='binary',
    n_estimators=100,
    early_stopping_rounds=10,
    verbosity=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='logloss',
    #verbose=0
)

# 输出特征重要性
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("LightGBM 特征重要性：\n", importances)
