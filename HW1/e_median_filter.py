import numpy as np
import cv2
import matplotlib.pyplot as plt

def image_median_filter(img, k_size=3):
    h, w = img.shape
    pad = k_size // 2

    # 使用 'reflect' 模式：鏡像反射邊緣，保留梯度資訊，適合去噪
    padded_img = np.pad(img, pad, mode='reflect')
    
    
    output = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            # 從填充後的影像中取出 k_size x k_size 的視窗
            window = padded_img[i : i + k_size, j : j + k_size]
            
            # 這裡直接使用 numpy 的 median 函數來計算中值，這樣可以確保結果的正確性和效率。
            output[i, j] = np.median(window)
    return output

def main():
    # 路徑設定
    input_noisy = "./output/noisy_lenna.png"
    output_result = "./output/result_e_median.png"      # 純結果圖 
    output_compare = "./output/compare_e_median.png"    # 比較圖 

    # 讀取雜訊圖
    img = cv2.imread(input_noisy, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("錯誤：找不到 'noisy_lenna.png'，請先執行 a_generate_noise.py")
        return

    img_processed = image_median_filter(img, 3)
    
    # 1. 儲存純結果圖
    cv2.imwrite(output_result, img_processed)

    # 2. 繪製並儲存比較圖
    plt.figure(figsize=(10, 5))
    # 左圖：處理前 (雜訊)
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Noisy Input")
    plt.axis('off')
    # 右圖：處理後 (濾波結果)
    plt.subplot(1, 2, 2)
    plt.imshow(img_processed, cmap='gray')
    plt.title("(e) Median Filter")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_compare, dpi=300)
    plt.show()
    print(f"(e) 完成：")
    print(f"    - 濾波結果已存: {output_result}")
    print(f"    - 比較圖已存:   {output_compare}")

if __name__ == "__main__":
    main()