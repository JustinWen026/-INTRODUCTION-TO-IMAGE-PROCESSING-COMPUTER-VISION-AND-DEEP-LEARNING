# -INTRODUCTION-TO-IMAGE-PROCESSING-COMPUTER-VISION-AND-DEEP-LEARNING
1141_影像處理、電腦視覺及深度學習概論HW2
影像處理、電腦視覺與深度學習 - 作業二

開發環境：Python 3.12 / OpenCV 4.10 / PySide6 (或是 PyQt5) 本作業涵蓋了電腦視覺中的幾何重建與深度學習分類任務，透過互動式 GUI 展現 3D 視覺與影像縫合技術。

功能展示 (Features)

1. 立體視覺 (Stereo Vision)深度圖計算 (Stereo Matching)：利用 cv2.StereoBM 演算法計算左右視圖的視差 (Disparity)，並將結果正規化至 0-255 以利視覺化。

互動式深度偵測：當點擊左圖的某個點時，程式會自動在右圖標註對應點，並根據公式 $Distance = \frac{baseline \times f}{disparity + doff}$ 計算出該點的實際深度（單位：mm）。

2. 影像對齊與拼接 (Image Alignment & Stitching)關鍵點偵測：使用 SIFT 演算法提取兩張影像的特徵點，並繪製出特徵分布圖。

特徵匹配：利用 cv2.BFMatcher 結合 KNN 演算法尋找最佳匹配對，並透過 cv2.drawMatches 展示配對結果。

影像拼接：計算 Homography 矩陣進行透視變換，將兩張具有重疊區域的圖片完美縫合為一場全景圖。

3. 深度學習 - VGG16 分類 (Deep Learning)模型架構：實作 VGG16 卷積神經網路，用於 CIFAR-10 數據集的影像分類任務。

資料增強 (Data Augmentation)：包含隨機水平翻轉與縮放，提升模型的泛化能力。

訓練與預測：展示訓練過程中的 Accuracy 與 Loss 曲線圖 。提供互動介面：上傳圖片後，模型會即時預測分類結果（如：貓、狗、船等）並顯示各類別的機率分布。

技術關鍵點SIFT (Scale-Invariant Feature Transform)：用於處理具有旋轉或縮放差異的影像匹配。R

ANSAC：在影像拼接過程中，用於過濾錯誤的匹配點以計算準確的投影矩陣。

Transfer Learning：(可選) 說明是否使用了預訓練模型來加速收斂。
