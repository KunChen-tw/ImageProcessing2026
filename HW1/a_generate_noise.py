import numpy as np 
import cv2

def add_sp_noise(image, salt_prob=0.15, pepper_prob=0.15):
    noisy_image = image.copy()
    h, w = image.shape[:2]
    num_pixels = h * w

    # 計算要放多少鹽/胡椒點
    num_salt = int(num_pixels * salt_prob)
    num_pepper = int(num_pixels * pepper_prob)

    if noisy_image.ndim == 2:  # grayscale
            salt_rows = np.random.randint(0, h, num_salt)
            salt_cols = np.random.randint(0, w, num_salt)
            pepper_rows = np.random.randint(0, h, num_pepper)
            pepper_cols = np.random.randint(0, w, num_pepper)

            noisy_image[salt_rows, salt_cols] = 255
            noisy_image[pepper_rows, pepper_cols] = 0

    elif noisy_image.ndim == 3:  # color, each channel independently
        c = noisy_image.shape[2]
        for ch in range(c):
            salt_rows = np.random.randint(0, h, num_salt)
            salt_cols = np.random.randint(0, w, num_salt)
            pepper_rows = np.random.randint(0, h, num_pepper)
            pepper_cols = np.random.randint(0, w, num_pepper)

            noisy_image[salt_rows, salt_cols, ch] = 255
            noisy_image[pepper_rows, pepper_cols, ch] = 0
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return noisy_image

    

def main():

       # 設定路徑
    input_path = "./output/Lenna.bmp"
    output_path = "./output/noisy_lenna.png"
    
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"無法讀取檔案：{input_path}（請確認路徑與檔名）")
        return
    
    Area=img[0:256,256:512]
    Area_img = add_sp_noise(Area, salt_prob=0.15, pepper_prob=0.15)
    noisy_img = img.copy()  # 建立一個副本來顯示噪聲效果
    noisy_img[0:256, 256:512] = Area_img # 直接修改原圖的一部分以顯示噪聲效果
   
    # 儲存結果供後續程式使用
    cv2.imwrite(output_path, noisy_img)
    print(f"(a) 完成：已產生雜訊影像並儲存為 '{output_path}'")
    print(f"    雜訊區域：右上角 [0:256, 256:512]")


    cv2.imshow("Original Image", img)
    cv2.imshow("Salt and Pepper Noise Image", noisy_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
