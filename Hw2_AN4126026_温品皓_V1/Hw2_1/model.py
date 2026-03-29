import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Q1: LeNet-5 Implementation [cite: 97]
class LeNet5(nn.Module):
    def __init__(self, activation='relu'):
        super(LeNet5, self).__init__()
        self.activation_type = activation
        
        # 根據 Spec Page 8 的架構圖
        # Input: 1x32x32 (MNIST resized)
        # C1: 6 filters, 5x5 kernel -> 6x28x28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # S2: Average Pooling 2x2 -> 6x14x14
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 16 filters, 5x5 kernel -> 16x10x10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # S4: Average Pooling 2x2 -> 16x5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # C5: 120 connections (Flatten後接 FC)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # F6: 84 connections
        self.fc2 = nn.Linear(120, 84)
        # Output: 10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        act = torch.sigmoid if self.activation_type == 'sigmoid' else F.relu
        
        x = self.conv1(x)
        x = act(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = act(x)
        x = self.pool2(x)
        
        x = x.view(-1, 16 * 5 * 5) # Flatten
        x = self.fc1(x)
        x = act(x)
        
        x = self.fc2(x)
        x = act(x)
        
        x = self.fc3(x)
        return x

# Q2: ResNet18 Modified 
class ResNet18_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet18_CIFAR, self).__init__()
        # 載入預設 ResNet18，不需 pretrained (因為要從頭練 CIFAR)
        self.model = models.resnet18(pretrained=False)
        
        # 修改 1: 將原本 7x7 stride 2 的 conv1 改為 3x3 stride 1
        # 因為 CIFAR-10 只有 32x32，原本的 conv1 會讓特徵圖變太小
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 修改 2: 移除 maxpool (原本在 conv1 之後)
        # 這裡我們透過覆寫 forward 函數或把 maxpool 替換成 Identity 來達成
        self.model.maxpool = nn.Identity()
        
        # 修改 3: 修改 Fully Connected Layer，輸出改為 10 類
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)