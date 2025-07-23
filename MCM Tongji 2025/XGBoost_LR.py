import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 1. 特征选择 ==========
data = pd.read_excel('data_nn.xlsx')
X = data.drop(columns=['BHR'])
y = data['BHR'].astype(int)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 特征聚类分析
corr_matrix = X.corr().abs()
distance_matrix = 1 - corr_matrix
condensed_dist = squareform(distance_matrix)
Z = linkage(condensed_dist, method='average')

threshold = 0.7
clusters = fcluster(Z, t=threshold, criterion='distance')

# 特征选择
selected_features = []
target_corrs = X.apply(lambda col: col.corr(y)).abs()

for cluster_id in np.unique(clusters):
    cluster_features = X.columns[clusters == cluster_id]
    best_feature = target_corrs[cluster_features].idxmax()
    selected_features.append(best_feature)

print(f"Selected features: {selected_features}")

# ========== 2. 模型构建 ==========
X_selected = X[selected_features]

# 初始化模型容器
xgb_models = []
lr_models = []
cv_scores = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # 训练XGBoost模型
    xgb = XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=fold
    )
    xgb.fit(X_train, y_train)
    xgb_models.append(xgb)
    
    # 训练最大熵模型（逻辑回归）
    lr = LogisticRegression(
        penalty='l2',
        C=0.1,
        solver='lbfgs',
        max_iter=1000,
        random_state=fold
    )
    lr.fit(X_train, y_train)
    lr_models.append(lr)
    
    # 集成预测验证集
    xgb_proba = xgb.predict_proba(X_val)[:, 1]
    lr_proba = lr.predict_proba(X_val)[:, 1]
    ensemble_proba = (xgb_proba + lr_proba) / 2
    
    val_auc = roc_auc_score(y_val, ensemble_proba)
    cv_scores.append(val_auc)
    print(f"Fold {fold+1} Ensemble AUC: {val_auc:.4f}")

# ========== 3. 模型评估 ==========
def ensemble_predict(X):
    xgb_probas = [model.predict_proba(X)[:, 1] for model in xgb_models]
    lr_probas = [model.predict_proba(X)[:, 1] for model in lr_models]
    avg_probas = (np.mean(xgb_probas, axis=0) + np.mean(lr_probas, axis=0)) / 2
    return (avg_probas >= 0.5).astype(int)

# 使用第一个fold的验证集作为示例
_, val_idx = next(cv.split(X_selected, y))
X_sample = X_selected.iloc[val_idx]
y_sample = y.iloc[val_idx]

# 评估指标
y_pred = ensemble_predict(X_sample)
print("\nFinal Evaluation:")
print(f"Average CV AUC: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
print(f"Accuracy: {accuracy_score(y_sample, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_sample, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_sample, ensemble_predict(X_sample)):.4f}")

# 可视化双模型特征分析
plt.figure(figsize=(12, 6))

# XGBoost特征重要性
plt.subplot(1, 2, 1)
xgb_importances = np.mean([model.feature_importances_ for model in xgb_models], axis=0)
sns.barplot(x=xgb_importances, y=selected_features)
plt.title('XGBoost Feature Importances')

# 逻辑回归系数绝对值
plt.subplot(1, 2, 2)
lr_coefs = np.mean([np.abs(model.coef_[0]) for model in lr_models], axis=0)
sns.barplot(x=lr_coefs, y=selected_features)
plt.title('Logistic Regression Coefficients (abs)')

plt.tight_layout()
plt.show()

# 混淆矩阵
cm = confusion_matrix(y_sample, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
