import numpy as np
import cv2


def image_log_filter(img):
    # LoG (Laplacian of Gaussian) 5x5 mask from textbook
    mask = np.array(
        [
            [1, 12, 24, 12, 1],
            [12, 58, 0, 58, 12],
            [24, 0, -424, 0, 24],
            [12, 58, 0, 58, 12],
            [1, 12, 24, 12, 1],
        ],
        dtype=np.float32,
    )

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("image_log_filter 只接受灰階、BGR 或 BGRA 影像")

    gray_f32 = gray.astype(np.float32)
    # 用 BORDER_REPLICATE 代替 BORDER_CONSTANT，避免邊界假邊緣
    response = cv2.filter2D(gray_f32, cv2.CV_32F, mask, borderType=cv2.BORDER_REPLICATE)

    # 取絕對值，把正負響應都變成亮邊
    response_abs = np.abs(response)
    
    # 線性拉伸到 0~255 便於顯示
    min_val = np.min(response_abs)
    max_val = np.max(response_abs)
    if max_val > min_val:
        display = (response_abs - min_val) * 255.0 / (max_val - min_val)
    else:
        display = np.zeros_like(response_abs)
    
    return response, display.astype(np.uint8)


def zero_crossing_detection(log_response, threshold=20):
    """
    偵測 Laplace 響應中的 Zero Crossing 邊緣（改進版本2 - 保留原始方式作為預設）
    
    原理：
    - 檢查每個像素及其 8 鄰域是否有符號變化（正→負 或 負→正）
    - 只有當變化幅度足夠大時才標記為邊緣
    
    Args:
        log_response: LoG 濾波的響應 (浮點數)
        threshold: T，邊緣幅度門檻值。越大邊緣點越少。建議試試 20~100
    
    Returns:
        edges: 二值影像，邊緣為 255，其他為 0
    """
    h, w = log_response.shape
    edges = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = log_response[i, j]
            
            # 檢查 8 個鄰域
            neighbors = [
                log_response[i-1, j-1], log_response[i-1, j], log_response[i-1, j+1],
                log_response[i,   j-1],                       log_response[i,   j+1],
                log_response[i+1, j-1], log_response[i+1, j], log_response[i+1, j+1]
            ]
            
            # 原始方式：檢查符號變化且變化幅度 >= threshold
            for neighbor in neighbors:
                if center * neighbor < 0:  # 正負號相反
                    if abs(center - neighbor) >= threshold:
                        edges[i, j] = 255
                        break
    
    return edges



def main():
    input1 = "./HW2/img/Lenna.bmp"
    # input2 = "./HW2/img/chessking.jpg"
    # input3 = "./HW2/img/Breakfast.bmp"
    output1 = "./HW2/output/b_log_result_lenna.png"
    # output2 = "./HW2/output/b_result_b_laplace_chessking.png"
    # output3 = "./HW2/output/b_result_b_laplace_breakfast.png"
    compare_output1 = "./HW2/output/b_log_compare_lenna.png"
    # compare_output2 = "./HW2/output/b_compare_b_laplace_chessking.png"
    # compare_output3 = "./HW2/output/b_compare_b_laplace_breakfast.png"

    img1 = cv2.imread(input1, cv2.IMREAD_COLOR)
    # img2 = cv2.imread(input2, cv2.IMREAD_COLOR)
    # img3 = cv2.imread(input3, cv2.IMREAD_COLOR)

    if img1 is None:
        print(f"無法讀取檔案：{input1}")
        return
    # if img2 is None:
    #     print(f"無法讀取檔案：{input2}")
    #     return
    # if img3 is None:
    #     print(f"無法讀取檔案：{input3}")
    #     return

    log_raw1, out1 = image_log_filter(img1)
    # log_raw2, out2 = image_log_filter(img2)
    # log_raw3, out3 = image_log_filter(img3)

    # 可調參數：LoG 響應正規化尺度，越大代表同一個 T 會偵測到更多邊緣
    detection_scale = 40.0

    # 測試多個 T 值
    threshold_values = [10, 20, 30]
    results = {}

    cv2.imwrite(output1, out1)
    # cv2.imwrite(output2, out2)
    # cv2.imwrite(output3, out3)

    # 保存 raw 和 abs 版本
    raw_output1 = output1.replace("_result_", "_raw_")
    abs_output1 = output1.replace("_result_", "_abs_")
    
    # Raw: 有符號響應（將負值移到 128 顯示中灰）
    raw_display = np.clip(log_raw1 + 128.0, 0, 255).astype(np.uint8)
    cv2.imwrite(raw_output1, raw_display)
    
    # Abs: 絕對值（out1 即為 abs 版拉伸圖）
    cv2.imwrite(abs_output1, out1)

    # 先把 LoG 響應做尺度正規化，讓課本常用的 T=10/20/30 更有區分度
    max_abs = np.max(np.abs(log_raw1))
    if max_abs > 0:
        log_for_detection = (log_raw1 / max_abs) * detection_scale
    else:
        log_for_detection = log_raw1.copy()

    # 對每個 T 值執行 Zero Crossing Edge Detection 並存檔
    for T in threshold_values:
        edges = zero_crossing_detection(log_for_detection, threshold=T)
        results[T] = edges
        
        edge_output = output1.replace("_result_", f"_edge_T{T}_")
        cv2.imwrite(edge_output, edges)
        
        edge_count = np.count_nonzero(edges)
        total_pixels = edges.size
        edge_percentage = (edge_count / total_pixels) * 100
        
        print(f"T={T:2d} - 邊緣檔案：{edge_output}，邊緣像素：{edge_count}，佔比：{edge_percentage:.2f}%")

    # 動態合併 threshold_values 中所有 T 值的邊緣結果
    out1_bgr = cv2.cvtColor(out1, cv2.COLOR_GRAY2BGR)
    # compare_parts = [img1, out1_bgr]  # 先放原圖和 LoG 絕對值
    compare_parts = [img1]

    for T in threshold_values:
        edges_bgr = cv2.cvtColor(results[T], cv2.COLOR_GRAY2BGR)
        compare_parts.append(edges_bgr)
    
    compare1 = np.hstack(compare_parts)
    # compare2 = np.hstack([img2, out2_bgr])
    # compare3 = np.hstack([img3, out3_bgr])

    cv2.imwrite(compare_output1, compare1)
    # cv2.imwrite(compare_output2, compare2)
    # cv2.imwrite(compare_output3, compare3)

    can_show = True
    try:
        cv2.imshow("Lenna Input | LoG Output", compare1)
        # cv2.imshow("Chessking Input | LoG Output", compare2)
        # cv2.imshow("Breakfast Input | LoG Output", compare3)
    except cv2.error:
        can_show = False

    # 統計基本資訊
    min_response = int(np.min(log_raw1))
    max_response = int(np.max(log_raw1))
    
    print(f"\nLoG Filter 完成：{output1}")
    print(f"LoG Raw 版本（有符號）：{raw_output1}")
    print(f"LoG Abs 版本（絕對值）：{abs_output1}")
    print(f"LoG 比較圖完成：{compare_output1}")
    print(f"LoG 原始響應範圍：{min_response} 到 {max_response}")
    print(f"LoG 邊緣偵測使用正規化尺度：[-{int(detection_scale)}, {int(detection_scale)}]")
    
    # 保存統計結果到文本檔案
    stats_output = output1.replace("_result_", "_stats_").replace(".png", ".txt")
    with open(stats_output, "w", encoding="utf-8") as f:
        f.write("=== LoG Edge Detection Statistics ===\n\n")
        f.write(f"Input Image: {input1}\n\n")
        f.write(f"LoG Response Range: {min_response} ~ {max_response}\n")
        f.write(f"Detection Scale: normalized to [-{detection_scale:.1f}, {detection_scale:.1f}]\n")
        f.write(f"Total Pixels: {results[10].size}\n\n")
        f.write("=== Zero Crossing Detection Results ===\n")
        for T in threshold_values:
            edges = results[T]
            edge_count = np.count_nonzero(edges)
            total_pixels = edges.size
            edge_percentage = (edge_count / total_pixels) * 100
            f.write(f"T={T:2d}: Edge Pixels={edge_count:7d}, Percentage={edge_percentage:6.2f}%\n")
    
    print(f"統計結果已保存：{stats_output}")
    
    if can_show:
        print("已顯示 input / output，按任意鍵關閉視窗")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("目前 OpenCV 無 GUI 視窗支援，已略過 imshow（結果已儲存到 output 資料夾）。")


if __name__ == "__main__":
    main()
