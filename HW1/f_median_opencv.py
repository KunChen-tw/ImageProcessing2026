import cv2
import matplotlib.pyplot as plt

def image_median_filter_cv(img, kernel_size=3):
    return cv2.medianBlur(img, kernel_size)

def main():
    # 路徑設定
    input_noisy = "./output/noisy_lenna.png"
    output_result = "./output/result_f_median_opencv.png"      # 純結果圖 
    output_compare = "./output/compare_f_median_opencv.png"    # 比較圖 

    # 讀取雜訊圖
    img = cv2.imread(input_noisy, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("錯誤：找不到 'noisy_lenna.png'，請先執行 a_generate_noise.py")
        return

    img_processed = image_median_filter_cv(img, 3)
    
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
    plt.title("(f) Median Filter (OpenCV)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_compare, dpi=300)
    plt.show()
    print(f"(f) 完成：")
    print(f"    - 濾波結果已存: {output_result}")
    print(f"    - 比較圖已存:   {output_compare}")

if __name__ == "__main__":
    main()