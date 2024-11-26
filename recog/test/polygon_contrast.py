#!/usr/bin/env python

import cv2
import numpy as np

# ポリゴンの頂点を格納するリスト
points = []

def mouse_callback(event, x, y, flags, param):
    """
    マウスイベントを処理してポリゴンを描画
    """
    global points, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:  # 左クリックで点を追加
        points.append((x, y))
        cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Draw Polygon", img_copy)

    elif event == cv2.EVENT_RBUTTONDOWN:  # 右クリックでポリゴン確定
        if len(points) > 2:
            cv2.polylines(img_copy, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.imshow("Draw Polygon", img_copy)

def apply_brightness_in_polygon(img, polygon):
    """
    ポリゴン内の明るさを強調
    """
    # マスクを作成
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon)], 255)

    # YUV色空間に変換
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # 明るさチャンネル（Yチャンネル）を取得
    y_channel = img_yuv[:, :, 0]

    # ポリゴン内のみヒストグラム均等化
    y_channel_eq = cv2.equalizeHist(y_channel)
    y_channel = np.where(mask == 255, y_channel_eq, y_channel)

    # Yチャンネルを更新してBGRに戻す
    img_yuv[:, :, 0] = y_channel
    img_result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_result

def main(image_path, output_path):
    global img_copy

    # 画像を読み込み
    img = cv2.imread(image_path)
    img_copy = img.copy()

    # ウィンドウの設定
    cv2.namedWindow("Draw Polygon")
    cv2.setMouseCallback("Draw Polygon", mouse_callback)

    print("左クリックでポリゴンの頂点を指定してください。右クリックで確定します。")
    while True:
        cv2.imshow("Draw Polygon", img_copy)
        key = cv2.waitKey(1)

        if key == 27:  # ESCキーで終了
            print("終了しました。")
            break

        if key == ord("p"):  # 'p'キーでポリゴン内の明るさ強調
            if len(points) > 2:
                print("ポリゴン内の明るさを強調します...")
                result = apply_brightness_in_polygon(img, points)
                cv2.imshow("Result", result)
                cv2.imwrite(output_path, result)
                print(f"結果を保存しました: {output_path}")
                break

    cv2.destroyAllWindows()

# 入力画像と出力画像のパス
input_image = "l0045.png"
output_image = "polygon_brightness_fixed_left.png"

if __name__ == "__main__":
    main(input_image, output_image)
