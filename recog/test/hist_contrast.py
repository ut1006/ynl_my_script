#!/usr/bin/env python

import cv2
import numpy as np

def adjust_gamma(image_path, output_path, gamma=1.5):
    # 画像を読み込み
    img = cv2.imread(image_path)
    
    # ガンマ補正用のルックアップテーブルを作成
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # ガンマ補正を適用
    img_gamma = cv2.LUT(img, table)
    
    # 保存
    cv2.imwrite(output_path, img_gamma)

# 入力画像と出力パス
input_image = "l0045.png"
output_image = "gammma0.png"

# ガンマ補正によるコントラスト強調
adjust_gamma(input_image, output_image, gamma=0.4)
print(f"Gamma-corrected image saved as {output_image}")
