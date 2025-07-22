import json
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# 自定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, json_file, base_path, transform=None):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.images = data['images']
        self.annotations = data['annotations']
        self.base_path = base_path  # 图片根目录
        self.transform = transform
        
        # 建立 image_id 到 category_id 的映射
        self.image_id_to_category = {ann['image_id']: ann['category_id'] for ann in self.annotations}
        
        # 获取图片路径和对应的 category_id
        self.image_paths = []
        self.labels = []
        for img in self.images:
            image_id = img['id']
            if image_id in self.image_id_to_category:
                # 不使用 JSON 中的 folder 字段，使用 base_path 拼接 file_name
                file_name = img['file_name']
                image_path = os.path.join(self.base_path, file_name)
                self.image_paths.append(image_path)
                self.labels.append(self.image_id_to_category[image_id])
        
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
def load_datasets(train_json, val_json, test_json, base_path):
    train_dataset = CustomImageDataset(train_json, base_path, transform=transform)
    val_dataset = CustomImageDataset(val_json, base_path, transform=transform)
    test_dataset = CustomImageDataset(test_json, base_path, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_classes = train_dataset.num_classes
    return train_loader, val_loader, test_loader, num_classes