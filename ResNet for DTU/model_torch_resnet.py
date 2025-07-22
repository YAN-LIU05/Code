import torch.nn as nn
from torchvision import models


# 初始化 ResNet 模型
def initialize_model(num_classes):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

