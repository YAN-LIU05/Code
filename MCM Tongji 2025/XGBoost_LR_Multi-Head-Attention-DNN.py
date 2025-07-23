import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Multiply, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def attention_block(inputs, name):
    """多头注意力机制"""
    # 注意力权重
    attention = Dense(1, activation='tanh')(inputs)
    attention = Activation('softmax', name=name)(attention)
    # 应用注意力权重
    attended = Multiply()([inputs, attention])
    return attended

def create_model(trial):
    """创建带注意力机制的DNN模型"""
    # 获取输入维度
    global input_dim
    input_dim = X_selected.shape[1]
    
    # 超参数搜索空间
    params = {
        'lr': trial.suggest_float('dnn_lr', 1e-4, 1e-2, log=True),
        'units1': trial.suggest_int('dnn_units1', 32, 128),
        'units2': trial.suggest_int('dnn_units2', 16, 64),
        'dropout1': trial.suggest_float('dnn_dropout1', 0.2, 0.5),
        'dropout2': trial.suggest_float('dnn_dropout2', 0.1, 0.3)
    }
    
    # 模型架构
    inputs = Input(shape=(input_dim,), dtype='float32')
    x = Dense(params['units1'], activation='relu')(inputs)
    x = Dropout(params['dropout1'])(x)
    x = attention_block(x, 'att1')
    x = Dense(params['units2'], activation='relu')(x)
    x = Dropout(params['dropout2'])(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=params['lr']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def objective(trial):
    """Optuna优化目标函数"""
    # 超参数搜索空间
    params = {
        'xgb_lr': trial.suggest_float('xgb_lr', 0.01, 0.3),
        'xgb_depth': trial.suggest_int('xgb_depth', 3, 10),
        'xgb_subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'xgb_colsample': trial.suggest_float('xgb_colsample', 0.6, 1.0),
        'lr_C': trial.suggest_float('lr_C', 0.1, 10.0, log=True)
    }
    
    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X_selected, y):
        # 训练XGBoost
        xgb = XGBClassifier(
            learning_rate=params['xgb_lr'],
            max_depth=params['xgb_depth'],
            subsample=params['xgb_subsample'],
            colsample_bytree=params['xgb_colsample'],
            use_label_encoder=False
        )
        xgb.fit(X_selected.iloc[train_idx], y.iloc[train_idx])
        
        # 训练Logistic Regression
        lr = LogisticRegression(
            C=params['lr_C'],
            max_iter=1000
        )
        lr.fit(X_selected.iloc[train_idx], y.iloc[train_idx])
        
        # 训练DNN
        dnn = create_model(trial)
        X_train = X_selected.iloc[train_idx].values.astype('float32')
        y_train = y.iloc[train_idx].values.astype('float32')
        X_val = X_selected.iloc[val_idx].values.astype('float32')
        y_val = y.iloc[val_idx].values.astype('float32')
        
        dnn.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(patience=10)],
            verbose=0
        )
        
        # 集成预测
        xgb_proba = xgb.predict_proba(X_selected.iloc[val_idx])[:,1]
        lr_proba = lr.predict_proba(X_selected.iloc[val_idx])[:,1]
        dnn_proba = dnn.predict(X_selected.iloc[val_idx].values.astype('float32')).flatten()
        
        avg_proba = (xgb_proba + lr_proba + dnn_proba) / 3
        score = roc_auc_score(y.iloc[val_idx], avg_proba)
        scores.append(score)
    
    return np.mean(scores)

def ensemble_predict(X):
    """集成模型预测"""
    # 直接使用已经标准化的数据，不需要再次转换
    X_scaled = X.values.astype('float32')
    
    xgb_probas = [m.predict_proba(X_scaled)[:,1] for m in final_xgb_models]
    lr_probas = [m.predict_proba(X_scaled)[:,1] for m in final_lr_models]
    dnn_probas = [m.predict(X_scaled, verbose=0).flatten() for m in final_dnn_models]
    
    avg_proba = (np.mean(xgb_probas, axis=0) + 
                np.mean(lr_probas, axis=0) + 
                np.mean(dnn_probas, axis=0)) / 3
    return (avg_proba >= 0.5).astype(int), avg_proba

from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping

def plot_roc_curve(y_true, y_proba, save_path=None):
    """绘制ROC曲线并计算AUC"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc_score(y_true, y_proba):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # 对角线
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('集成模型ROC曲线')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_importance(final_xgb_models, feature_names, save_path=None):
    """绘制XGBoost模型的平均特征重要性"""
    feature_importance = np.mean([model.feature_importances_ for model in final_xgb_models], axis=0)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_names)
    plt.title('XGBoost平均特征重要性')
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_probability_distribution(y_true, y_proba, save_path=None):
    """绘制预测概率的分布"""
    plt.figure(figsize=(8, 6))
    sns.histplot(y_proba[y_true == 0], color='blue', label='类别0', alpha=0.5)
    sns.histplot(y_proba[y_true == 1], color='red', label='类别1', alpha=0.5)
    plt.xlabel('预测概率')
    plt.ylabel('计数')
    plt.title('按类别划分的预测概率分布')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_dnn_learning_curves(model, X_train, y_train, X_val, y_val, save_path=None):
    """绘制DNN模型的训练和验证损失曲线"""
    history = model.fit(
        X_train.astype('float32'),
        y_train.astype('float32'),
        epochs=100,
        batch_size=32,
        validation_data=(X_val.astype('float32'), y_val.astype('float32')),
        callbacks=[EarlyStopping(patience=10)],
        verbose=0
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.title('DNN学习曲线')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return history

# 主程序
if __name__ == "__main__":
    # 1. 数据加载和预处理
    print("加载数据...")
    data = pd.read_excel('data_nn.xlsx')
    X = data.drop('BHR', axis=1)
    y = data['BHR']
    
    # 2. 特征缩放
    print("特征缩放...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 3. 特征选择
    print("特征选择...")
    # 这里可以添加特征选择逻辑
    # 使用XGBoost进行特征重要性评估
    feature_selector = XGBClassifier(use_label_encoder=False)
    feature_selector.fit(X_scaled, y)
    
    # 获取特征重要性
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_selector.feature_importances_
    })
    importances = importances.sort_values('importance', ascending=False)
    
    # 选择前8个最重要的特征
    top_features = importances['feature'].head(8).tolist()
    X_selected = X_scaled[top_features]
    
    print("\n选择的特征:")
    for i, feature in enumerate(top_features, 1):
        print(f"{i}. {feature}")
    
    # 4. 超参数优化
    print("开始超参数优化...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    # 获取最佳参数
    best_params = study.best_params
    print("\n最佳参数:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # 5. 模型训练
    print("\n开始训练最终模型...")
    final_xgb_models = []
    final_lr_models = []
    final_dnn_models = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
        print(f"\n训练第 {fold+1} 折...")
        
        # 训练XGBoost
        print("训练XGBoost模型...")
        xgb = XGBClassifier(
            learning_rate=best_params['xgb_lr'],
            max_depth=best_params['xgb_depth'],
            subsample=best_params['xgb_subsample'],
            colsample_bytree=best_params['xgb_colsample'],
            use_label_encoder=False
        )
        xgb.fit(X_selected.iloc[train_idx], y.iloc[train_idx])
        final_xgb_models.append(xgb)
        
        # 训练Logistic Regression
        print("训练逻辑回归模型...")
        lr = LogisticRegression(
            C=best_params['lr_C'],
            max_iter=1000
        )
        lr.fit(X_selected.iloc[train_idx], y.iloc[train_idx])
        final_lr_models.append(lr)
        
        # 训练DNN
        print("训练DNN模型...")
        dnn = create_model(study.best_trial)
        X_train = X_selected.iloc[train_idx].values.astype('float32')
        y_train = y.iloc[train_idx].values.astype('float32')
        X_val = X_selected.iloc[val_idx].values.astype('float32')
        y_val = y.iloc[val_idx].values.astype('float32')
        
        dnn.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(patience=10)],
            verbose=1
        )
        final_dnn_models.append(dnn)
        print(f"第 {fold+1} 折训练完成")
    
    print("\n所有模型训练完成，开始评估...")
    
    # 6. 模型评估
    _, val_idx = next(cv.split(X_selected, y))
    X_val = X_selected.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    y_pred, y_proba = ensemble_predict(X_val)
    print("\n最终模型性能:")
    print(f"AUC: {roc_auc_score(y_val, y_proba):.4f}")
    print(f"准确率: {accuracy_score(y_val, y_pred):.4f}")
    print(f"F1分数: {f1_score(y_val, y_pred):.4f}")
    
    # 7. 可视化
    print("\n生成可视化...")
    # 在主程序的模型评估部分添加可视化
    print("\nPerforming visualization...")
    # ROC curve
    plot_roc_curve(y_val, y_proba, save_path='roc_curve.png')

    # Confusion matrix
    plot_confusion_matrix(y_val, y_pred, save_path='confusion_matrix.png')

    # Feature importance
    plot_feature_importance(final_xgb_models, X_selected.columns, save_path='feature_importance.png')

    # Probability distribution
    plot_probability_distribution(y_val, y_proba, save_path='probability_distribution.png')

    # DNN learning curves (using the first DNN model as an example)
    plot_dnn_learning_curves(
        final_dnn_models[0],
        X_selected.iloc[train_idx].values,
        y.iloc[train_idx].values,
        X_selected.iloc[val_idx].values,
        y.iloc[val_idx].values,
        save_path='dnn_learning_curve.png'
    )