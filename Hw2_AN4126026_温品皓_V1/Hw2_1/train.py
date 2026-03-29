import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import LeNet5, ResNet18_CIFAR

def train_model(model_type='lenet', activation='relu', epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # --- 1. 資料準備 ---
    if model_type == 'lenet':
        # MNIST transform [cite: 101]
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        model = LeNet5(activation=activation).to(device)
        
    else: # resnet
        # CIFAR-10 transform
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        model = ResNet18_CIFAR().to(device)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 用來紀錄數值畫圖用
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0

    # --- 2. Training Loop ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # 紀錄數據
        epoch_acc = 100 * correct_val / total_val
        train_losses.append(running_loss / len(trainloader))
        val_losses.append(val_loss / len(valloader))
        train_accs.append(100 * correct_train / total_train)
        val_accs.append(epoch_acc)

        print(f"Epoch {epoch+1}/{epochs}, Val Acc: {epoch_acc:.2f}%")

        # 存最佳權重
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_name = f"weights_{model_type}_{activation}.pth" if model_type=='lenet' else "resnet18.pth"
            torch.save(model.state_dict(), save_name)

    # --- 3. 畫圖並儲存 ---
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(f'loss_{model_type}_{activation}.jpg')
    
    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig(f'acc_{model_type}_{activation}.jpg')
    print("Training Done!")

if __name__ == "__main__":
    # 你需要自己手動取消註解來跑不同的訓練
    # train_model('lenet', 'sigmoid', 20)
    # train_model('lenet', 'relu', 20)
    train_model('resnet', 'relu', 30) # ResNet 通常需要久一點