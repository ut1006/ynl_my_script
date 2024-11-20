import rosbag
import cv2
from cv_bridge import CvBridge
import os
import argparse

def extract_images_from_rosbag(bag_file, output_dir, image_topic):
    # 画像を保存するディレクトリを作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # CvBridgeの初期化
    bridge = CvBridge()

    # 最後に画像を保存した時刻の初期/zedm/zed_node/left/image_rect_color化
    last_saved_time = None
    image_count = 0
    start_time = None
    end_time = None

    # rosbagファイルを開いて画像を抽出
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            # 最初のメッセージのタイムスタンプを取得
            if start_time is None:
                start_time = t

            # 現在のメッセージのタイムスタンプを秒単位で取得
            current_time = t.secs + t.nsecs * 1e-9

            # 最後のメッセージのタイムスタンプを更新
            end_time = t

            # 最初の画像を保存するか、1秒経過しているかを確認
            if last_saved_time is None or (current_time - last_saved_time >= 1.0):
                try:
                    # ROSメッセージをOpenCV形式の画像に変換
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                    
                    # タイムスタンプをファイル名に利用
                    timestamp = str(t.to_nsec())
                    file_name = os.path.join(output_dir, f"{timestamp}.png")
                    
                    # 画像をPNG形式で保存
                    cv2.imwrite(file_name, cv_image)
                    print(f"Saved {file_name}")
                    
                    # 保存した画像の枚数をカウント
                    image_count += 1

                    # 最後に保存した時刻を更新
                    last_saved_time = current_time
                
                except Exception as e:
                    print(f"Failed to save image: {e}")

    # ROSBAGの総再生時間を計算 (秒単位)
    if start_time and end_time:
        total_time = (end_time.secs + end_time.nsecs * 1e-9) - (start_time.secs + start_time.nsecs * 1e-9)
        print(f"Total duration of the rosbag: {total_time:.2f} seconds")
    
    # 保存した画像の枚数を出力
    print(f"Total number of saved images: {image_count}")

if __name__ == "__main__":
    # 引数の設定
    parser = argparse.ArgumentParser(description="Convert images from rosbag to PNG format every 1 second.")
    parser.add_argument("bag_file", type=str, help="Path to the input rosbag file")
    parser.add_argument("output_dir", type=str, help="Directory to save the output PNG images")
    parser.add_argument("--image_topic", type=str, default="/zedm/zed_node/left/image_rect_color", help="Image topic name in the rosbag")
    args = parser.parse_args()

    # 関数を実行
    extract_images_from_rosbag(args.bag_file, args.output_dir, args.image_topic)
