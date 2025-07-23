import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from deal_with_random_forest import top_8, df

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class BHRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BHRPredictor(nn.Module):
    def __init__(self, input_size):
        super(BHRPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

def train_model():
 # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print("选择的特征:", top_8)
    
    # 准备数据
    X = df[top_8]
    y = df['BHR']
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = BHRDataset(X_train, y_train.values)
    test_dataset = BHRDataset(X_test, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型并移至GPU
    model = BHRPredictor(input_size=len(top_8)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练过程
    epochs = 100
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    print("开始训练...")
    for epoch in trange(epochs, desc='训练进度'):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        for batch_X, batch_y in train_bar:
            # 将数据移至GPU
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_bar.set_postfix({'train_loss': f'{loss.item():.4f}'})
            
        # 验证
        model.eval()
        total_test_loss = 0
        test_bar = tqdm(test_loader, desc='验证', leave=False)
        with torch.no_grad():
            for batch_X, batch_y in test_bar:
                # 将数据移至GPU
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                total_test_loss += loss.item()
                test_bar.set_postfix({'test_loss': f'{loss.item():.4f}'})
                
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_test_loss = total_test_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'best_bhr_predictor.pth')
        
        if (epoch + 1) % 10 == 0:
            tqdm.write(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.title('模型训练过程中的损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    
    # 模型评估
    model.eval()
    with torch.no_grad():
        # 获取预测概率和标签
        test_probs = model(torch.FloatTensor(X_test).to(device))
        test_probs = test_probs.cpu().numpy()
        test_predictions = test_probs > 0.5
        
        # 计算各项指标
        accuracy = (test_predictions.flatten() == y_test.values).mean()
        print(f'测试集准确率: {accuracy:.4f}')
        
        # 打印分类报告（包含精确率、召回率、F1分数）
        print("\n分类报告:")
        print(classification_report(y_test.values, test_predictions.flatten(), 
                                 target_names=['Class 0', 'Class 1']))
        
        # 绘制混淆矩阵
        plot_confusion_matrix(y_test.values, test_predictions.flatten())
        
        # 绘制ROC曲线
        plot_roc_curve(y_test.values, test_probs.flatten())
        
        # 保存最终模型
        torch.save(model.state_dict(), 'final_bhr_predictor.pth')
        
    return {
        'accuracy': accuracy,
        'predictions': test_predictions,
        'probabilities': test_probs,
        'true_labels': y_test.values
    }


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path='roc_curve.png'):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    train_model()