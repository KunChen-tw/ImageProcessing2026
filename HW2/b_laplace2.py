import cv2
import numpy as np
import matplotlib.pyplot as plt

input1 = "./HW2/img/Lenna.bmp"

# 1. 讀取影像並轉為灰階
img = cv2.imread(input1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 高斯模糊 (重要！先去除雜訊，否則過零點會非常多且雜亂)
# ksize=(5,5) 是模糊核大小，sigma=1.0 是標準差
# blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

# 3. 拉普拉斯運算
# cv2.CV_64F 表示使用 64-bit float 來儲存結果，因為拉普拉斯結果會有負值
# laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# 4. 過零點偵測與門檻值處理 (Zero Crossing + Threshold)
# 這裡我們手動實作邏輯，因為 OpenCV 沒有直接的 "ZeroCrossing" 函數

# 設定門檻值：只有絕對值大於此數值的變化才被視為邊緣
threshold = 10 

# 計算絕對值並轉換為 uint8 (為了顯示)
abs_laplacian = np.uint8(np.absolute(laplacian))

# 尋找過零點 (Zero Crossing)
# 原理：檢查像素與其右方、下方的像素，若符號不同則為過零點
# 我們利用 np.sign 來判斷正負號
sign_map = np.sign(laplacian)
# 建立一個位移後的 sign map 來比較相鄰像素
sign_shift_right = np.roll(sign_map, -1, axis=1)
sign_shift_down = np.roll(sign_map, -1, axis=0)

# 如果相鄰像素符號不同，則乘積為負數
zero_crossings = (sign_map * sign_shift_right < 0) | (sign_map * sign_shift_down < 0)

# 5. 套用門檻值
# 只有當「是過零點」且「拉普拉斯絕對值大於門檻值」時，才保留
# 注意：這裡我們檢查過零點附近的強度是否夠大
edge_map = np.zeros_like(abs_laplacian)
# 將布尔值轉換為索引
y_coords, x_coords = np.where(zero_crossings)

for y, x in zip(y_coords, x_coords):
    # 邊界檢查
    if y < abs_laplacian.shape[0] and x < abs_laplacian.shape[1]:
        # 檢查該點的強度是否大於門檻值
        if abs_laplacian[y, x] > threshold:
            edge_map[y, x] = 255

# 6. 顯示結果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title("Original Gray Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(abs_laplacian, cmap='gray')
plt.title("Absolute Laplacian (Strength)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edge_map, cmap='gray')
plt.title(f"Zero Crossing (Threshold={threshold})")
plt.axis('off')

plt.tight_layout()
plt.show()