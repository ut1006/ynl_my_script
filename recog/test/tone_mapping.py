#!/usr/bin/env python

import cv2
import numpy as np

def apply_global_tone_mapping(image_path, output_path, gamma=2.2):
    """
    グローバルトーンマッピングを適用して画像を明るく調整
    :param image_path: 入力画像のパス
    :param output_path: 出力画像の保存パス
    :param gamma: ガンマ値（大きくするほど明るくなる）
    """
    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not read the image from {image_path}")

    # 画像を正規化 (0-1の範囲にスケーリング)
    normalized_image = image / 255.0

    # ガンマ補正を適用
    tone_mapped_image = np.power(normalized_image, 1 / gamma)

    # スケールを戻して保存
    output_image = (tone_mapped_image * 255).astype(np.uint8)
    cv2.imwrite(output_path, output_image)
    print(f"Tone-mapped image saved to {output_path}")

if __name__ == "__main__":
    input_image_path = "l0045.png"
    output_image_path = "l0045_tone_mapped.png"

    try:
        apply_global_tone_mapping(input_image_path, output_image_path, gamma=2.2)
    except ValueError as e:
        print(e)
