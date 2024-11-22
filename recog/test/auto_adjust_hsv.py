#!/usr/bin/env python

import cv2
import numpy as np

def auto_adjust_saturation_brightness(image_path, output_path, target_brightness=180, target_saturation=150):
    """
    自動的に画像の彩度と明度を調整する
    :param image_path: 入力画像のパス
    :param output_path: 出力画像の保存パス
    :param target_brightness: 目標の明度平均値
    :param target_saturation: 目標の彩度平均値
    """
    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not read the image from {image_path}")

    # BGRからHSVに変換
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSVチャンネルを分割
    h, s, v = cv2.split(hsv_image)

    # 現在の明度と彩度の平均値
    current_brightness = v.mean()
    current_saturation = s.mean()

    # 明度と彩度のスケーリング係数を計算
    brightness_factor = target_brightness / current_brightness
    saturation_factor = target_saturation / current_saturation

    # スケーリングの適用（クリップで安全な範囲に制限）
    v = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)
    s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)

    # 調整後のHSV画像を再構築
    adjusted_hsv_image = cv2.merge((h, s, v))

    # BGRに戻して保存
    adjusted_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, adjusted_image)
    print(f"Adjusted image saved to {output_path}")

if __name__ == "__main__":
    input_image_path = "r0045.png"
    output_image_path = "r0045_auto_adjusted.png"

    try:
        auto_adjust_saturation_brightness(input_image_path, output_image_path)
    except ValueError as e:
        print(e)
