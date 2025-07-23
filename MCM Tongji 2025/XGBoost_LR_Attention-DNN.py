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
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ========== 1. 注意力机制实现 ==========
def attention_block(inputs, name):
    """特征注意力机制模块"""
    # 注意力权重计算
    attention = Dense(1, activation='tanh')(inputs)
    attention = Activation('softmax')(attention)
    
    # 应用注意力权重
    weighted = Multiply(name=name)([inputs, attention])
    return weighted

# ========== 2. 贝叶斯优化目标函数 ==========
def create_model(trial):
    """创建带注意力机制的DNN模型"""
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
    # 定义所有模型的超参数空间
    xgb_params = {
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.3),
        'max_depth': trial.suggest_int('xgb_depth', 3, 7),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 0.9)
    }
    
    lr_params = {
        'C': trial.suggest_float('lr_C', 0.01, 1.0, log=True)
    }
    
    # 交叉验证
    scores = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, val_idx in cv.split(X_selected, y):
        X_train = X_selected.iloc[train_idx].values.astype('float32')
        y_train = y.iloc[train_idx].values.astype('float32')
        X_val = X_selected.iloc[val_idx].values.astype('float32')
        y_val = y.iloc[val_idx].values.astype('float32')
        
        # 训练XGBoost
        xgb = XGBClassifier(**xgb_params, use_label_encoder=False)
        xgb.fit(X_train, y_train, verbose=False)
        
        # 训练Logistic Regression
        lr = LogisticRegression(**lr_params, max_iter=1000)
        lr.fit(X_train, y_train)
        
        # 训练DNN
        dnn = create_model(trial)
        dnn.fit(X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[EarlyStopping(patience=5)],
                verbose=0)
        
        # 集成预测
        xgb_proba = xgb.predict_proba(X_val)[:,1]
        lr_proba = lr.predict_proba(X_val)[:,1]
        dnn_proba = dnn.predict(X_val, verbose=0).flatten()
        ensemble_proba = (xgb_proba + lr_proba + dnn_proba) / 3
        
        scores.append(roc_auc_score(y_val, ensemble_proba))
    
    return np.mean(scores)

# ========== 3. 优化执行与模型训练 ==========
# 数据准备
data = pd.read_excel("data_nn.xlsx")
X = data.drop(columns=['BHR'])
y = data['BHR'].astype(int)

scaler = StandardScaler()
X_selected = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
input_dim = X_selected.shape[1]

# 贝叶斯优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=3600)

# 使用最佳参数训练最终模型
best_params = study.best_params
print("Best parameters:", best_params)

# 初始化最终模型容器
final_xgb_models = []
final_lr_models = []
final_dnn_models = []

# 完整训练流程
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
    # 训练XGBoost
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
    lr = LogisticRegression(
        C=best_params['lr_C'],
        max_iter=1000
    )
    lr.fit(X_selected.iloc[train_idx], y.iloc[train_idx])
    final_lr_models.append(lr)
    
    # 训练DNN
    dnn = create_model(study.best_trial)
    dnn.fit(
        X_selected.iloc[train_idx].values,
        y.iloc[train_idx],
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=10)],
        verbose=0
    )
    final_dnn_models.append(dnn)

# ========== 4. 集成评估 ==========
def ensemble_predict(X):
    X_scaled = scaler.transform(X).astype('float32')
    xgb_probas = [m.predict_proba(X_scaled)[:,1] for m in final_xgb_models]
    lr_probas = [m.predict_proba(X_scaled)[:,1] for m in final_lr_models]
    dnn_probas = [m.predict(X_scaled, verbose=0).flatten() for m in final_dnn_models]
    
    avg_proba = (np.mean(xgb_probas, axis=0) + 
                np.mean(lr_probas, axis=0) + 
                np.mean(dnn_probas, axis=0)) / 3
    return (avg_proba >= 0.5).astype(int), avg_proba

# 示例评估（使用第一个验证集）
_, val_idx = next(cv.split(X_selected, y))
X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

y_pred, y_proba = ensemble_predict(X_val)
print("\nFinal Model Performance:")
print(f"AUC: {roc_auc_score(y_val, y_proba):.4f}")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")

# ========== 5. 可视化分析 ==========
# 注意力权重可视化
sample_data = X_selected.sample(1).values
attention_layer = final_dnn_models[0].get_layer('att1')
attention_model = Model(inputs=final_dnn_models[0].input,
                        outputs=attention_layer.output)
attention_weights = attention_model.predict(sample_data).flatten()

plt.figure(figsize=(10,6))
sns.barplot(x=attention_weights, y=X.columns)
plt.title('Feature Attention Weights')
plt.show()
