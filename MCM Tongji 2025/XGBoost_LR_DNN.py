from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from itertools import combinations

def train_and_evaluate_models(X, y):
    """训练和评估多个模型"""
    # 创建5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 初始化结果存储
    results = {
        'XGBoost': {'accuracy': [], 'auc': []},
        'LogisticRegression': {'accuracy': [], 'auc': []},
        'DNN': {'accuracy': [], 'auc': []}
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"\n第 {fold} 折验证结果:")
        
        # XGBoost
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_val)
        xgb_prob = xgb.predict_proba(X_val)[:, 1]
        results['XGBoost']['accuracy'].append(accuracy_score(y_val, xgb_pred))
        results['XGBoost']['auc'].append(roc_auc_score(y_val, xgb_prob))
        
        # 逻辑回归
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_val)
        lr_prob = lr.predict_proba(X_val)[:, 1]
        results['LogisticRegression']['accuracy'].append(accuracy_score(y_val, lr_pred))
        results['LogisticRegression']['auc'].append(roc_auc_score(y_val, lr_prob))
        
        # DNN
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 训练DNN时不打印进度
        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        dnn_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int)
        dnn_prob = model.predict(X_val, verbose=0)
        results['DNN']['accuracy'].append(accuracy_score(y_val, dnn_pred))
        results['DNN']['auc'].append(roc_auc_score(y_val, dnn_prob))
        
        # 打印当前折的结果
        print("\nXGBoost:")
        print(f"Accuracy: {results['XGBoost']['accuracy'][-1]:.4f}")
        print(f"AUC: {results['XGBoost']['auc'][-1]:.4f}")
        
        print("\n逻辑回归:")
        print(f"Accuracy: {results['LogisticRegression']['accuracy'][-1]:.4f}")
        print(f"AUC: {results['LogisticRegression']['auc'][-1]:.4f}")
        
        print("\nDNN:")
        print(f"Accuracy: {results['DNN']['accuracy'][-1]:.4f}")
        print(f"AUC: {results['DNN']['auc'][-1]:.4f}")
    
    # 计算并打印平均结果
    print("\n=== 五折交叉验证平均结果 ===")
    for model_name in results:
        mean_accuracy = np.mean(results[model_name]['accuracy'])
        mean_auc = np.mean(results[model_name]['auc'])
        std_accuracy = np.std(results[model_name]['accuracy'])
        std_auc = np.std(results[model_name]['auc'])
        
        print(f"\n{model_name}:")
        print(f"Average Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Average AUC: {mean_auc:.4f} ± {std_auc:.4f}")

def perform_clustering(X):
    """执行特征聚类"""
    corr_matrix = X.corr().abs()
    distance_matrix = 1 - corr_matrix
    Z = linkage(squareform(distance_matrix), method='average')
    clusters = fcluster(Z, t=0.7, criterion='distance')
    return clusters

def generate_interactions(X, clusters):
    """基于特征聚类生成交互项"""
    interaction_features = []
    for cluster_id in np.unique(clusters):
        cluster_cols = X.columns[clusters == cluster_id].tolist()
        if len(cluster_cols) >= 2:
            for col1, col2 in combinations(cluster_cols, 2):
                interaction_name = f"{col1}&{col2}"
                interaction_features.append(pd.Series(
                    X[col1] * X[col2],
                    name=interaction_name
                ))
    
    if interaction_features:
        return pd.concat(interaction_features, axis=1)
    else:
        return pd.DataFrame(index=X.index)

def select_features_from_clusters(X, y, clusters):
    """从每个聚类中选择最佳特征"""
    selected_features = []
    original_features = X.columns[:len(clusters)]
    target_corrs = X[original_features].apply(lambda col: col.corr(y)).abs()
    
    for cluster_id in np.unique(clusters):
        cluster_features = original_features[clusters == cluster_id]
        if len(cluster_features) > 0:
            best_feature = target_corrs[cluster_features].idxmax()
            selected_features.append(best_feature)
    
    return selected_features

def main():
    # 读取并预处理数据
    data = pd.read_excel("data_nn.xlsx")
    X = data.drop(columns=['BHR'])
    y = data['BHR'].astype(int)

    # 在聚类之前打印特征数量
    print("原始特征总数:", len(X.columns))

    # 执行特征聚类
    clusters = perform_clustering(X)
    print("聚类数组长度:", len(clusters))

    # 生成交互特征
    X_interact = generate_interactions(X, clusters)
    X_enhanced = pd.concat([X, X_interact], axis=1)
    print("增强后的特征总数:", len(X_enhanced.columns))

    # 特征选择（在增强特征空间进行）
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_enhanced), columns=X_enhanced.columns)

    # 从每个聚类中选择最佳特征
    selected_original_features = select_features_from_clusters(X_scaled, y, clusters)
    print(f"\n从聚类中选择的原始特征 ({len(selected_original_features)}):")
    print(selected_original_features)

    # 添加相关的交互特征
    interaction_features = [col for col in X_interact.columns 
                          if any(feat in col for feat in selected_original_features)]
    final_features = selected_original_features + interaction_features
    print(f"\n最终选择的特征 (包括交互项) ({len(final_features)}):")
    print(final_features)

    # 使用选定的特征
    X_final = X_scaled[final_features]
    
    # 训练和评估模型
    print("\n开始模型训练和评估...")
    train_and_evaluate_models(X_final, y)

if __name__ == "__main__":
    main() 