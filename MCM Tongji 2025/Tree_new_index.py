import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_excel('data.xlsx')

# 处理EOS列缺失值
if 'EOS' in data.columns and data['EOS'].isnull().sum() > 0:
    # 准备特征（排除第一列和EOS列）
    features = data.columns.difference([data.columns[0], 'EOS'])
    
    # 分割数据
    known_eos = data[data['EOS'].notnull()]
    unknown_eos = data[data['EOS'].isnull()]
    
    if not unknown_eos.empty and not known_eos.empty:
        # 训练决策树模型
        dt_eos = DecisionTreeRegressor()
        dt_eos.fit(known_eos[features], known_eos['EOS'])
        
        # 预测并填充缺失值
        predicted = dt_eos.predict(unknown_eos[features])
        data.loc[data['EOS'].isnull(), 'EOS'] = predicted

# 处理BHR列后的异常值
if 'BHR=1' in data.columns:
    # 获取BHR后的列
    bhr_index = data.columns.get_loc('BHR=1')
    post_columns = data.columns[bhr_index+1:]
    
    for col in post_columns:
        if data[col].dtype in [np.float64, np.int64]:
            # 检测异常值（使用IQR方法）
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5*IQR
            upper_bound = Q3 + 1.5*IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            if not outliers.empty:
                # 准备特征（排除第一列和当前列）
                features = data.columns.difference([data.columns[0], col])
                
                # 训练模型
                dt_outlier = DecisionTreeRegressor()
                dt_outlier.fit(data.loc[~data.index.isin(outliers.index)][features],
                             data.loc[~data.index.isin(outliers.index), col])
                
                # 替换异常值
                predictions = dt_outlier.predict(outliers[features])
                data.loc[outliers.index, col] = predictions

# 合成新指标（使用PCA）
# 准备数据（排除第一列）
processed_data = data.drop(data.columns[0], axis=1)

# 标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(processed_data)

# 生成主成分
pca = PCA(n_components=1)
principal_component = pca.fit_transform(scaled_data)
data['New_Feature'] = principal_component

# 计算相关系数
if 'BHR' in data.columns:
    correlation = data['New_Feature'].corr(data['BHR'])
    print(f"Pearson相关系数: {correlation:.4f}")
else:
    print("数据中不存在BHR列")

# 保存结果（可选）
data.to_excel('data_nn.xlsx', index=True)
