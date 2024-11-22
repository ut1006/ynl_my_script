#!/usr/bin/env python

import cv2
import numpy as np

def get_brightness_saturation_range(image_path):
    """
    画像内の明度（V）と彩度（S）の最大値と最小値を取得する
    :param image_path: 入力画像のパス
    :return: (brightness_min, brightness_max, saturation_min, saturation_max)
    """
    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not read the image from {image_path}")

    # BGRからHSVに変換
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSVチャンネルを分割
    _, s, v = cv2.split(hsv_image)

    # 明度と彩度の最大・最小値を計算
    brightness_min, brightness_max = v.min(), v.max()
    saturation_min, saturation_max = s.min(), s.max()

    return brightness_min, brightness_max, saturation_min, saturation_max

if __name__ == "__main__":
    input_image_path = "l0045.png"

    # 明度と彩度の範囲を取得
    try:
        brightness_min, brightness_max, saturation_min, saturation_max = get_brightness_saturation_range(input_image_path)
        print(f"明度 (V) の範囲: 最小値 = {brightness_min}, 最大値 = {brightness_max}")
        print(f"彩度 (S) の範囲: 最小値 = {saturation_min}, 最大値 = {saturation_max}")
    except ValueError as e:
        print(e)
