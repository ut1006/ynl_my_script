from __future__ import print_function, division
import sys
sys.path.append('core')



# python run_model.py --restore_ckpt /path/to/your/checkpoint.pth


import argparse
import time
import logging
import cv2
import numpy as np
import torch
from tqdm import tqdm
from dlnr import DLNR, autocast
from utils.utils import InputPadder


@torch.no_grad()
def process_images(model, img_left_path, img_right_path, iters=32, mixed_prec=False):
    """
    指定された画像ペアを処理し、結果を返す。

    Args:
        model: 学習済みモデル。
        img_left_path (str): 左画像のパス。
        img_right_path (str): 右画像のパス。
        iters (int): 推論の繰り返し回数。
        mixed_prec (bool): Mixed Precisionを使用するか。

    Returns:
        flow_pr (torch.Tensor): 推論結果の光フロー。
    """
    # 画像の読み込みと前処理
    image1 = cv2.imread(img_left_path)
    image2 = cv2.imread(img_right_path)

    if image1 is None or image2 is None:
        raise ValueError("画像の読み込みに失敗しました。パスを確認してください。")

    # BGRからRGBに変換し、テンソルに変換
    image1 = torch.from_numpy(image1[..., ::-1].transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 255.0
    image2 = torch.from_numpy(image2[..., ::-1].transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 255.0

    # パディング
    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    # 推論
    with autocast(enabled=mixed_prec):
        start = time.time()
        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        end = time.time()

    # パディングを除去
    flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

    print(f"推論完了: 処理時間 {end - start:.3f}s")
    return flow_pr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default='/your_path/dlnr.pth')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=True)
    parser.add_argument('--valid_iters', type=int, default=10, help='number of flow-field updates during forward pass')
    args = parser.parse_args()

    # モデルのセットアップ
    model = torch.nn.DataParallel(DLNR(args), device_ids=[0])
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info("Done loading checkpoint")

    model.cuda()
    model.eval()

    # 入力画像パス
    img_left = "l0045.png"
    img_right = "r0045.png"

    # 推論
    flow = process_images(model, img_left, img_right, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    # 結果の保存または表示
    np.save("flow.npy", flow.numpy())  # 結果を保存
    print("推論結果を flow.npy に保存しました。")
