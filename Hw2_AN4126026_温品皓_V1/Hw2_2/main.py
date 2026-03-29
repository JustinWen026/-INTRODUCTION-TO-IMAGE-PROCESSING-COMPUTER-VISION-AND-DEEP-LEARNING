import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# PyQt5 Imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGroupBox, 
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

# 引用我們定義好的模型
try:
    from model import LeNet5, ResNet18_CIFAR
    try:
        from torchsummary import summary
    except ImportError:
        summary = None
except ImportError:
    print("Error: Could not import 'model.py'. Please check if the file exists.")
    sys.exit(1)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_image = None
        
        # 初始化模型變數
        self.lenet = None
        self.resnet = None
        
        # 初始化 UI
        self.initUI()
        
        # 載入模型
        self.load_models()

    def initUI(self):
        self.setWindowTitle('Hw2 - Deep Learning GUI')
        self.setGeometry(100, 100, 1000, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- 左邊控制面板 ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_widget.setFixedWidth(250)
        
        # Q1 Group
        group1 = QGroupBox('1. LeNet-5 (MNIST)')
        layout1 = QVBoxLayout()
        self.btn_1_load = QPushButton('Load Image')
        self.btn_1_load.clicked.connect(self.load_image)
        layout1.addWidget(self.btn_1_load)
        
        self.btn_1_1 = QPushButton('1.1 Show Architecture')
        self.btn_1_1.clicked.connect(self.show_lenet_structure)
        layout1.addWidget(self.btn_1_1)
        
        self.btn_1_2 = QPushButton('1.2 Show Acc Loss')
        self.btn_1_2.clicked.connect(self.show_lenet_loss_acc)
        layout1.addWidget(self.btn_1_2)
        
        self.btn_1_3 = QPushButton('1.3 Predict')
        self.btn_1_3.clicked.connect(self.inference_mnist)
        layout1.addWidget(self.btn_1_3)
        group1.setLayout(layout1)
        left_layout.addWidget(group1)
        
        # Q2 Group
        group2 = QGroupBox('2. ResNet18 (CIFAR-10)')
        layout2 = QVBoxLayout()
        self.btn_2_1 = QPushButton('2.1 Load Image')
        self.btn_2_1.clicked.connect(self.load_image)
        layout2.addWidget(self.btn_2_1)
        
        self.btn_2_2 = QPushButton('2.2 Show Architecture')
        self.btn_2_2.clicked.connect(self.show_resnet_structure)
        layout2.addWidget(self.btn_2_2)
        
        self.btn_2_3 = QPushButton('2.3 Show Acc Loss')
        self.btn_2_3.clicked.connect(self.show_resnet_loss_acc)
        layout2.addWidget(self.btn_2_3)
        
        self.btn_2_4 = QPushButton('2.4 Inference')
        self.btn_2_4.clicked.connect(self.inference_cifar)
        layout2.addWidget(self.btn_2_4)
        group2.setLayout(layout2)
        left_layout.addWidget(group2)
        
        left_layout.addStretch()
        main_layout.addWidget(left_widget)
        
        # --- 右邊顯示區域 ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.label_img_display = QLabel("Image Display Area")
        self.label_img_display.setAlignment(Qt.AlignCenter)
        self.label_img_display.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
        self.label_img_display.setFixedSize(500, 500)
        right_layout.addWidget(self.label_img_display)
        
        self.label_prediction = QLabel("Prediction Result:")
        self.label_prediction.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(self.label_prediction)
        
        main_layout.addWidget(right_widget)

    def load_models(self):
        # 取得目前工作路徑，方便除錯
        current_dir = os.getcwd()
        print(f"Current Working Directory: {current_dir}")

        # --- Q1 Loading ---
        try:
            path = os.path.join('model', 'Weight_Relu.pth')
            print(f"[Q1] Trying to load: {os.path.abspath(path)}")
            
            if os.path.exists(path):
                self.lenet = LeNet5('relu').to(self.device)
                self.lenet.load_state_dict(torch.load(path, map_location=self.device))
                self.lenet.eval() # 設定為 eval 模式，避免 inference 時報錯
                print(" -> Q1 Model Loaded Successfully.")
            else:
                print(" -> Q1 Model file NOT found. (Normal if running Q2)")
                self.lenet = None
        except Exception as e:
            print(f" -> Q1 Loading Error: {e}")
            self.lenet = None

        # --- Q2 Loading ---
        try:
            path = os.path.join('model', 'weight.pth')
            print(f"[Q2] Trying to load: {os.path.abspath(path)}")
            
            if os.path.exists(path):
                self.resnet = ResNet18_CIFAR().to(self.device)
                self.resnet.load_state_dict(torch.load(path, map_location=self.device))
                self.resnet.eval() # 設定為 eval 模式
                print(" -> Q2 Model Loaded Successfully.")
            else:
                print(" -> Q2 Model file NOT found. (Normal if running Q1)")
                self.resnet = None
        except Exception as e:
            print(f" -> Q2 Loading Error: {e}")
            self.resnet = None

    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', '.', "Image Files (*.png *.jpg *.bmp)")
        if filename:
            try:
                self.loaded_image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Error reading file: {e}")
                return

            if self.loaded_image is None:
                self.label_prediction.setText("Failed to load image!")
                return
                
            img_rgb = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2RGB)
            h, w, c = img_rgb.shape
            bytes_per_line = c * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.label_img_display.setPixmap(pixmap.scaled(self.label_img_display.size(), Qt.KeepAspectRatio))
            self.label_prediction.setText("Image Loaded.")

    # --- Q1 Functions ---
    def show_lenet_structure(self):
        # 即使模型沒載入，也顯示架構給助教看
        model = self.lenet if self.lenet else LeNet5('relu').to(self.device)
        print("-" * 30)
        print("LeNet-5 Architecture:")
        if summary:
            summary(model, (1, 32, 32))
        else:
            print(model)
        print("-" * 30)

    def show_lenet_loss_acc(self):
        img_sigmoid = cv2.imread('Loss&Acc_Sigmoid.jpg')
        img_relu = cv2.imread('Loss&Acc_Relu.jpg')
        
        if img_sigmoid is None or img_relu is None:
            QMessageBox.warning(self, "Error", "Images not found.\nNeed 'Loss&Acc_Sigmoid.jpg' & 'Loss&Acc_Relu.jpg'")
            return
            
        if img_sigmoid.shape[1] != img_relu.shape[1]:
            img_relu = cv2.resize(img_relu, (img_sigmoid.shape[1], img_relu.shape[0]))
        combined_img = np.vstack((img_sigmoid, img_relu))
        cv2.imshow("Q1 Comparison", combined_img)

    def inference_mnist(self):
        if self.loaded_image is None: 
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
        
        if self.lenet is None:
            QMessageBox.critical(self, "Error", "Q1 Model not loaded!\nCheck 'model/Weight_Relu.pth'")
            return
        
        img_gray = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (32, 32))
        img_gray = cv2.bitwise_not(img_gray)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        input_tensor = transform(img_gray).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                output = self.lenet(input_tensor)
                prob = F.softmax(output, dim=1)
                pred_cls = torch.argmax(prob, dim=1).item()
            
            self.label_prediction.setText(f"Prediction: {pred_cls}")
            self.show_histogram(prob.cpu().numpy()[0], [str(i) for i in range(10)], "Probability (MNIST)")
        except Exception as e:
            print(f"Inference Error: {e}")

    # --- Q2 Functions ---
    def show_resnet_structure(self):
        model = self.resnet if self.resnet else ResNet18_CIFAR().to(self.device)
        print("-" * 30)
        print("ResNet18 Architecture:")
        print(model)
        print("-" * 30)

    def show_resnet_loss_acc(self):
        img_resnet = cv2.imread('Loss&Acc.jpg')
        if img_resnet is None:
            QMessageBox.warning(self, "Error", "Image 'Loss&Acc.jpg' not found.")
            return
        cv2.imshow("Q2 ResNet Acc & Loss", img_resnet)

    def inference_cifar(self):
        if self.loaded_image is None: 
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
            
        if self.resnet is None:
            # 這裡就是你原本跳錯誤視窗的地方，代表 weight.pth 沒讀到
            QMessageBox.critical(self, "Error", "Q2 Model not loaded!\nCheck 'model/weight.pth'")
            return
        
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        img_rgb = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (32, 32))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        input_tensor = transform(img_resized).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                output = self.resnet(input_tensor)
                prob = F.softmax(output, dim=1)
                max_prob, pred_idx = torch.max(prob, dim=1)
                max_prob = max_prob.item()
                pred_idx = pred_idx.item()
            
            threshold = 0.7
            if max_prob < threshold:
                result_text = "Prediction: Others"
            else:
                result_text = f"Prediction: {classes[pred_idx]} ({max_prob:.2f})"
                
            self.label_prediction.setText(result_text)
            self.show_histogram(prob.cpu().numpy()[0], classes, "Probability (CIFAR-10)")
        except Exception as e:
             print(f"Inference Error: {e}")

    def show_histogram(self, probs, labels, title):
        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, probs)
        plt.title(title)
        plt.ylabel('Probability')
        plt.ylim(0, 1.1)
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{prob:.2f}', 
                     ha='center', va='bottom')
        plt.show()

def main():
    # 強制將工作目錄切換到 main.py 所在的資料夾
    if hasattr(sys, '_MEIPASS'):
        os.chdir(sys._MEIPASS)
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()