import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


from data import load_datasets


choice = input("choose model: 1 for resnet of torch, 2 for resnet-18, 3 for resnet-50\n")
if choice == '1':
    # 使用torchvision的ResNet模型
    from model_torch_resnet import initialize_model
elif choice == '2':
    # 使用自定义ResNet-18模型
    from model_resnet18 import initialize_model
elif choice == '3':
    # 使用自定义ResNet-50模型
    from model_resnet50 import initialize_model



# 验证模型
def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    return val_loss, val_accuracy, all_preds, all_labels

# 测试模型
def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = accuracy_score(all_labels, all_preds)
    return test_accuracy, all_preds, all_labels

# 可视化训练过程
def plot_training_curves(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_curves_{choice}.png')

# 绘制混淆矩阵
def plot_confusion_matrix(true_labels, pred_labels, num_classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix_{choice}.png')

# 训练模型
def train_model(model, train_loader, val_loader, num_epochs=5):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_accuracy = 0.0
    best_model_path = 'best_resnet_classifier.pth'
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        val_loss, val_accuracy, _, _ = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model with Val Accuracy: {best_val_accuracy:.4f}')
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_accuracies)
    
    return model, best_model_path


# 主函数
def main(train_json, val_json, test_json, base_path):
    # 加载数据
    train_loader, val_loader, test_loader, num_classes = load_datasets(train_json, val_json, test_json, base_path)
    
    # 初始化模型
    model = initialize_model(num_classes)
    
    # 训练模型
    model, best_model_path = train_model(model, train_loader, val_loader, num_epochs=20)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    
    # 测试模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_accuracy, test_preds, test_labels = test_model(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print('Classification Report:')
    print(classification_report(test_labels, test_preds))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(test_labels, test_preds, num_classes)
    
    # 保存最终模型
    torch.save(model.state_dict(), f'final_resnet_classifier_{choice}.pth')

if __name__ == '__main__':
    # JSON 文件路径
    train_json = r'annotations\train1024-s.json'
    val_json = r'annotations\val1024-s.json'
    test_json = r'annotations\test1024-s.json'
    # 图片根目录
    base_path = r'Nordtank_dealed'
    
    main(train_json, val_json, test_json, base_path)