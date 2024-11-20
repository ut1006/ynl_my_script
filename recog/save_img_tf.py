import rospy
import cv2
import os
import csv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from glob import glob
import tf

# グローバル変数
left_image_rect = None
right_image_rect = None
bridge = CvBridge()
listener = None

# 出力ディレクトリを作成
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 連番を取得する関数
def get_next_filename_count(directory):
    files = glob(os.path.join(directory, "*"))
    if not files:
        return 1  # 1から始める
    latest_dir = max(files, key=lambda x: int(os.path.basename(x)))
    latest_num = int(os.path.basename(latest_dir))
    return latest_num + 1

# 左カメラのrect画像を受け取るコールバック
def left_image_rect_callback(msg):
    global left_image_rect
    left_image_rect = bridge.imgmsg_to_cv2(msg, "bgr8")

# 右カメラのrect画像を受け取るコールバック
def right_image_rect_callback(msg):
    global right_image_rect
    right_image_rect = bridge.imgmsg_to_cv2(msg, "bgr8")

# TF（カメラ姿勢）を取得してCSVに保存する関数
def save_tf_data(count, dir_name):
    try:
        # base_linkからzedm_camera_centerまでのTFを取得
        (trans, rot) = listener.lookupTransform('base_link', 'zedm_left_camera_frame', rospy.Time(0))

        # CSVファイルに姿勢情報を保存
        tf_filename = os.path.join(dir_name, f"tf{count:04d}.csv")  # 保存先を画像と同じディレクトリに変更
        with open(tf_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # ヘッダーの行
            writer.writerow(['Translation X', 'Translation Y', 'Translation Z',
                             'Rotation X', 'Rotation Y', 'Rotation Z', 'Rotation W'])
            # 平行移動と回転の値をコンマ区切りで書き込む 全体をメートルに治した。
            writer.writerow([trans[0] , trans[1] , trans[2] , rot[0], rot[1], rot[2], rot[3]])
        print(f"Saved TF data to {tf_filename}")
        print(trans,rot)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        print("Failed to get transform")


# エンターキーが押されたら画像を保存
def save_images():
    while not rospy.is_shutdown():
        input_key = input("Press Enter to save images or 'q' to quit: ")

        if input_key.lower() == 'q':
            print("Exiting...")
            break

        if left_image_rect is not None and right_image_rect is not None:
            # 新しい連番のディレクトリを作成
            count = get_next_filename_count(output_dir)
            dir_name = os.path.join(output_dir, f"{count:04d}")
            os.makedirs(dir_name)

            # ファイル名を作成
            left_rect_filename = os.path.join(dir_name, f"l{count:04d}.png")
            right_rect_filename = os.path.join(dir_name, f"r{count:04d}.png")

            # 画像を保存
            cv2.imwrite(left_rect_filename, left_image_rect)
            cv2.imwrite(right_rect_filename, right_image_rect)

            # TF（カメラ姿勢）を保存
            save_tf_data(count, dir_name)  # ディレクトリパスを渡す

            print(f"Saved {left_rect_filename} and {right_rect_filename}")
        else:
            print("No rectified images received yet.")

if __name__ == "__main__":
    rospy.init_node('stereo_image_saver')

    # TFリスナーを初期化
    listener = tf.TransformListener()

    # 左右カメラのrect画像をサブスクライブ
    rospy.Subscriber("/zedm/zed_node/left/image_rect_color", Image, left_image_rect_callback)
    rospy.Subscriber("/zedm/zed_node/right/image_rect_color", Image, right_image_rect_callback)

    try:
        save_images()
    except rospy.ROSInterruptException:
        pass
