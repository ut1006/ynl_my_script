#!/usr/bin/env python

import cv2

def apply_histogram_equalization(image_path, output_path):
    """
    明度（Vチャンネル）にヒストグラム正規化を適用
    :param image_path: 入力画像のパス
    :param output_path: 出力画像の保存パス
    """
    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not read the image from {image_path}")

    # BGRからHSVに変換
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSVチャンネルを分割
    h, s, v = cv2.split(hsv_image)

    # Vチャンネル（明度）にヒストグラム正規化を適用
    v_equalized = cv2.equalizeHist(v)

    # 正規化後のHSV画像を再構築
    equalized_hsv_image = cv2.merge((h, s, v_equalized))

    # BGRに戻して保存
    equalized_image = cv2.cvtColor(equalized_hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, equalized_image)
    print(f"Equalized image saved to {output_path}")

if __name__ == "__main__":
    input_image_path = "l0045.png"
    output_image_path = "l0045_hist_eq.png"

    try:
        apply_histogram_equalization(input_image_path, output_image_path)
    except ValueError as e:
        print(e)
