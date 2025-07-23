import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==== 1. 数据集定义 ====
class BHRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==== 2. 模型定义 ====
class BHRPredictor(nn.Module):
    def __init__(self, input_size):
        super(BHRPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ==== 3. 特征选择函数 ====
def select_features(df, plot_heatmap=True, top_k=8):
    if 'FEV3' in df.columns and 'FVC' in df.columns:
        df['FEV3-FEV1/FVC'] = (df['FEV3'] - df['FEV1']) / df['FVC']
    
    corr_matrix = df.corr()['BHR'].abs().sort_values(ascending=False)
    corr_matrix = corr_matrix.drop('BHR')
    top_features = corr_matrix.head(top_k).index.tolist()

    if plot_heatmap:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[top_features + ['BHR']].corr(), annot=True, cmap='coolwarm')
        plt.title(f"与 BHR 最相关的 Top {top_k} 特征热力图")
        plt.tight_layout()
        plt.show()

    return top_features

# ==== 4. 模型训练函数 ====
def train_model(df):
    top_8 = select_features(df, plot_heatmap=False)
    print("选择的特征:", top_8)

    X = df[top_8]
    y = df['BHR']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 应用PCA降维
    pca = PCA(n_components=min(6, len(top_8)))
    X_pca = pca.fit_transform(X_scaled)
    print("PCA后累计解释方差比:", np.cumsum(pca.explained_variance_ratio_))

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.3, random_state=42
    )

    train_dataset = BHRDataset(X_train, y_train.values)
    test_dataset = BHRDataset(X_test, y_test.values)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BHRPredictor(input_size=X_pca.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')

    print("开始训练...")
    for epoch in trange(100, desc='训练进度'):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                total_test_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_test_loss = total_test_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch [{epoch+1}/100], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    # 可视化损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.title('模型训练过程中的损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve_pca.png')

    # 准确率评估
    model.eval()
    with torch.no_grad():
        test_predictions = model(torch.FloatTensor(X_test))
        test_predictions = (test_predictions.numpy() > 0.5).astype(int)
        accuracy = (test_predictions.flatten() == y_test.values).mean()
        print(f'测试集准确率: {accuracy:.4f}')

    torch.save(model.state_dict(), 'bhr_predictor_optimized.pth')

# ==== 5. 主函数入口 ====
if __name__ == '__main__':
    #from deal_with_random_forest import df
    df = pd.read_excel('data_nn.xlsx')
    df = df.dropna()  # 简单处理缺失值
    train_model(df)
