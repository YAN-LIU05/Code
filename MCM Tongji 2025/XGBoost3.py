import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 特征选择 ==========
# 读取数据
data = pd.read_excel('data2.xlsx')
X = data.drop(columns=['BHR'])  # 假设BHR是目标变量
y = data['BHR'].astype(int)

# 数据预处理
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 特征聚类分析
corr_matrix = X.corr().abs()
distance_matrix = 1 - corr_matrix
condensed_dist = squareform(distance_matrix)
Z = linkage(condensed_dist, method='average')

# 自动确定聚类数量（基于距离阈值）
threshold = 0.7  # 可调整的聚类阈值
clusters = fcluster(Z, t=threshold, criterion='distance')

# 特征选择策略：每个簇中选择与目标相关性最高的特征
selected_features = []
target_corrs = X.apply(lambda col: col.corr(y)).abs()

for cluster_id in np.unique(clusters):
    cluster_features = X.columns[clusters == cluster_id]
    best_feature = target_corrs[cluster_features].idxmax()
    selected_features.append(best_feature)

print(f"Selected features: {selected_features}")

# ========== 2. 模型构建 ==========
X_selected = X[selected_features]
models = []
cv_scores = []

# 交叉验证训练
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=fold
    )
    
    model.fit(X_train, y_train)
    models.append(model)
    
    # 验证集评估
    val_pred = model.predict(X_val)
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
    cv_scores.append(val_auc)
    print(f"Fold {fold+1} AUC: {val_auc:.4f}")


from sklearn.metrics import roc_curve

# 绘制ROC曲线
def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC 曲线 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='随机猜测')
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('ROC 曲线')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制训练损失曲线
def plot_learning_curves():
    plt.figure(figsize=(10, 6))
    
    # 存储每个折的训练和验证损失
    train_losses = []
    val_losses = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
        X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model = XGBClassifier(
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=fold
        )
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='logloss',
            verbose=False
        )
        
        results = model.evals_result()
        train_losses.append(results['validation_0']['logloss'])
        val_losses.append(results['validation_1']['logloss'])
    
    # 计算平均损失
    mean_train_loss = np.mean(train_losses, axis=0)
    mean_val_loss = np.mean(val_losses, axis=0)
    std_train_loss = np.std(train_losses, axis=0)
    std_val_loss = np.std(val_losses, axis=0)
    
    epochs = range(1, len(mean_train_loss) + 1)
    
    # 绘制损失曲线及其标准差范围
    plt.plot(epochs, mean_train_loss, 'b-', label='训练损失')
    plt.fill_between(epochs, 
                    mean_train_loss - std_train_loss,
                    mean_train_loss + std_train_loss,
                    alpha=0.1, color='b')
    
    plt.plot(epochs, mean_val_loss, 'r-', label='验证损失')
    plt.fill_between(epochs,
                    mean_val_loss - std_val_loss,
                    mean_val_loss + std_val_loss,
                    alpha=0.1, color='r')
    
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True)
    plt.show()

# ========== 3. 模型评估 ==========
# 集成预测
def ensemble_predict(X):
    probas = np.mean([model.predict_proba(X)[:,1] for model in models], axis=0)
    return (probas >= 0.5).astype(int)

# 整体性能评估（使用第一个fold的验证集作为示例）
X_sample = X_selected.iloc[cv.split(X_selected, y).__next__()[1]]
y_sample = y.iloc[cv.split(X_selected, y).__next__()[1]]
y_pred = ensemble_predict(X_sample)

# 评估指标
print("\nModel Evaluation:")
print(f"Average CV AUC: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
print(f"Accuracy: {accuracy_score(y_sample, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_sample, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_sample, ensemble_predict(X_sample)):.4f}")

# 特征重要性可视化
feature_importances = np.mean([model.feature_importances_ for model in models], axis=0)
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances, y=selected_features)
plt.title('Feature Importances')
plt.show()

# 混淆矩阵
cm = confusion_matrix(y_sample, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\n绘制ROC曲线...")
y_proba = np.mean([model.predict_proba(X_sample)[:,1] for model in models], axis=0)
plot_roc_curve(y_sample, y_proba)

print("\n绘制学习曲线...")
plot_learning_curves()
