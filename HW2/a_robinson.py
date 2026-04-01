import numpy as np
import cv2


def image_robinson_filter(img):
    masks = [
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32),
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32),
        np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32),
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32),
        np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float32),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
        np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32),
    ]

    def robinson_single_channel(channel):
        h, w = channel.shape
        padded = np.pad(channel.astype(np.float32), 1, mode="reflect")
        response = np.zeros((h, w), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                window = padded[i : i + 3, j : j + 3]
                values = [np.sum(window * m) for m in masks]
                response[i, j] = np.max(values)

        return response

    if img.ndim == 2:
        response = robinson_single_channel(img)
    elif img.ndim == 3 and img.shape[2] == 3:
        channels = cv2.split(img)
        responses = [robinson_single_channel(ch) for ch in channels]
        response = np.max(np.stack(responses, axis=0), axis=0)
    elif img.ndim == 3 and img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        channels = cv2.split(bgr)
        responses = [robinson_single_channel(ch) for ch in channels]
        response = np.max(np.stack(responses, axis=0), axis=0)
    else:
        raise ValueError("image_robinson_filter 只接受灰階、BGR 或 BGRA 影像")

    # 線性拉伸到 0~255 便於顯示
    min_val = np.min(response)
    max_val = np.max(response)
    if max_val > min_val:
        response = (response - min_val) * 255.0 / (max_val - min_val)
    else:
        response.fill(0)

    return response.astype(np.uint8)



def main():
    input1 = "./HW2/img/Lenna.bmp"
    input2 = "./HW2/img/chessking.jpg"
    input3 = "./HW2/img/Breakfast.bmp"
    output1 = "./HW2/output/result_a_robinson_lenna.png"
    output2 = "./HW2/output/result_a_robinson_chessking.png"
    output3 = "./HW2/output/result_a_robinson_breakfast.png"
    compare_output1 = "./HW2/output/compare_a_robinson_lenna.png"
    compare_output2 = "./HW2/output/compare_a_robinson_chessking.png"
    compare_output3 = "./HW2/output/compare_a_robinson_breakfast.png"

    img1 = cv2.imread(input1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(input2, cv2.IMREAD_COLOR)
    img3 = cv2.imread(input3, cv2.IMREAD_COLOR)

    if img1 is None:
        print(f"無法讀取檔案：{input1}")
        return
    if img2 is None:
        print(f"無法讀取檔案：{input2}")
        return
    if img3 is None:
        print(f"無法讀取檔案：{input3}")
        return

    out1 = image_robinson_filter(img1)
    out2 = image_robinson_filter(img2)
    out3 = image_robinson_filter(img3)

    cv2.imwrite(output1, out1)
    cv2.imwrite(output2, out2)
    cv2.imwrite(output3, out3)

    # 顯示 input / output：將灰階輸出轉成 BGR 方便並排比較
    out1_bgr = cv2.cvtColor(out1, cv2.COLOR_GRAY2BGR)
    out2_bgr = cv2.cvtColor(out2, cv2.COLOR_GRAY2BGR)
    out3_bgr = cv2.cvtColor(out3, cv2.COLOR_GRAY2BGR)
    compare1 = np.hstack([img1, out1_bgr])
    compare2 = np.hstack([img2, out2_bgr])
    compare3 = np.hstack([img3, out3_bgr])

    cv2.imwrite(compare_output1, compare1)
    cv2.imwrite(compare_output2, compare2)
    cv2.imwrite(compare_output3, compare3)

    cv2.imshow("Lenna Input | Robinson Output", compare1)
    cv2.imshow("Chessking Input | Robinson Output", compare2)
    cv2.imshow("Breakfast Input | Robinson Output", compare3)

    print(f"Robinson Filter 完成：{output1}")
    print(f"Robinson Filter 完成：{output2}")
    print(f"Robinson Filter 完成：{output3}")
    print(f"Robinson 比較圖完成：{compare_output1}")
    print(f"Robinson 比較圖完成：{compare_output2}")
    print(f"Robinson 比較圖完成：{compare_output3}")
    print("已顯示 input / output，按任意鍵關閉視窗")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
