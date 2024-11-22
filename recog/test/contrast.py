#!/usr/bin/env python

import cv2

def adjust_brightness_and_save(input_path, output_path, alpha, beta):
    # 画像の読み込み
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Cannot load image from {input_path}")
        return

    # コントラストと明るさの変更
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 調整後の画像を保存
    cv2.imwrite(output_path, adjusted_image)
    print(f"Adjusted image saved to {output_path}")

if __name__ == "__main__":
    # 入力画像パス
    input_path = 'r0045.png'
    # 保存先画像パス
    output_path = 'r0045_ad.png'
    # コントラストと明るさの調整パラメータ
    alpha = 1  # コントラスト倍率
    beta = 20    # 明るさ調整値

    adjust_brightness_and_save(input_path, output_path, alpha, beta)
