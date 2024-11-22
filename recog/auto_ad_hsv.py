#!/usr/bin/env python

import os
import cv2
import numpy as np

def auto_adjust_saturation_brightness(image, target_brightness=180, target_saturation=150):
    """
    自動的に画像の彩度と明度を調整する
    :param image: 入力画像
    :param target_brightness: 目標の明度平均値
    :param target_saturation: 目標の彩度平均値
    :return: 調整後の画像
    """
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

    # BGRに戻して返す
    adjusted_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)
    return adjusted_image

def process_stereo_image_pair(input_dir, output_dir, target_brightness=180, target_saturation=150):
    """
    ステレオ画像ペアに対して、左右両方の画像を調整し保存する
    :param input_dir: 入力ディレクトリ
    :param output_dir: 出力ディレクトリ
    :param target_brightness: 目標の明度平均値
    :param target_saturation: 目標の彩度平均値
    """
    # ディレクトリ内の画像ファイルを取得
    for dir_name in os.listdir(input_dir):
        dir_path = os.path.join(input_dir, dir_name)
        if os.path.isdir(dir_path):
            left_image_path = os.path.join(dir_path, f'l{dir_name}.png')
            right_image_path = os.path.join(dir_path, f'r{dir_name}.png')

            # 画像の読み込み
            left_image = cv2.imread(left_image_path)
            right_image = cv2.imread(right_image_path)

            if left_image is None or right_image is None:
                print(f"Error: Could not read one or both images in {dir_name}. Skipping...")
                continue

            # 左右画像に対して調整を適用
            adjusted_left = auto_adjust_saturation_brightness(left_image, target_brightness, target_saturation)
            adjusted_right = auto_adjust_saturation_brightness(right_image, target_brightness, target_saturation)

            # 調整後の画像を保存
            adjusted_left_path = os.path.join(output_dir, dir_name, f'l{dir_name}_ad.png')
            adjusted_right_path = os.path.join(output_dir, dir_name, f'r{dir_name}_ad.png')

            os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)

            cv2.imwrite(adjusted_left_path, adjusted_left)
            cv2.imwrite(adjusted_right_path, adjusted_right)

            print(f"Processed and saved: {adjusted_left_path}, {adjusted_right_path}")

if __name__ == "__main__":
    input_dir = "output"  # 入力ディレクトリ
    output_dir = "output_adjusted"  # 出力ディレクトリ

    # ステレオ画像ペアに対する調整処理を実行
    process_stereo_image_pair(input_dir, output_dir)
