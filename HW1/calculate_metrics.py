# calculate_metrics.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def calculate_psnr(original, processed):
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

def calculate_ssim(original, processed):
    # score 是相似度的平均值
    score, _ = ssim(original, processed, full=True)
    return score

def main():
    output_path = "./output"
    original_path = "./img/Lenna.bmp"
    
    # 定義要比較的文件列表
    # 格式: (描述, 檔案名稱)
    files_to_check = [
        ("Max Filter", "result_b_max.png"),
        ("Min Filter", "result_c_min.png"),
        # 全距濾波器通常不拿來算與原圖的相似度，因為它是差異圖，這裡可選是否加入
        ("Range Filter", "result_d_range.png"), 
        ("Median", "result_e_median.png"),
        ("Median (OpenCV)", "result_f_median_opencv.png")
    ]
    
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print("錯誤：找不到原始圖片 Lenna.bmp")
        return

    print(f"{'Method':<20} | {'PSNR (dB)':<10} | {'SSIM':<10}")
    print("-" * 45)

    for name, filename in files_to_check:
        filepath = os.path.join(output_path, filename)
        if not os.path.exists(filepath):
            print(f"{name}: 檔案不存在")
            continue
            
        processed = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        # 確保尺寸一致 (以防萬一)
        if processed.shape != original.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
            
        psnr_val = calculate_psnr(original, processed)
        ssim_val = calculate_ssim(original, processed)
        
        print(f"{name:<20} | {psnr_val:>10.2f} | {ssim_val:>10.4f}")

if __name__ == "__main__":
    main()