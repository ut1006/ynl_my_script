#!/usr/bin/env python

import cv2
import numpy as np

def adjust_brightness_saturation(image_path, output_path, brightness_factor=1.5, saturation_factor=1.5):
    """
    明度と彩度を調整する関数
    :param image_path: 入力画像のパス
    :param output_path: 出力画像の保存パス
    :param brightness_factor: 明度のスケーリング係数
    :param saturation_factor: 彩度のスケーリング係数
    """
    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image from {image_path}")
        return

    # BGRからHSVに変換
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSVチャンネルを分割
    h, s, v = cv2.split(hsv_image)

    # 明度と彩度の最大・最小値を計算
    brightness_min, brightness_max = v.min(), v.max()
    saturation_min, saturation_max = s.min(), s.max()
    print(f"明度 (V) の範囲: 最小値 = {brightness_min}, 最大値 = {brightness_max}")
    print(f"彩度 (S) の範囲: 最小値 = {saturation_min}, 最大値 = {saturation_max}")
   
    # 彩度と明度を調整
    s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)
    v = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)

    # HSVを再結合
    adjusted_hsv_image = cv2.merge((h, s, v))

    # HSVからBGRに変換して保存
    adjusted_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, adjusted_image)
    print(f"Adjusted image saved to {output_path}")

if __name__ == "__main__":
    input_image_path = "l0045.png"
    output_image_path = "l0045_ad.png"
    adjust_brightness_saturation(input_image_path, output_image_path)
