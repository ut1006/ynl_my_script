#!/usr/bin/env python

import rospy
import tf
from aerial_robot_msgs.msg import FlightNav, PoseControlPid
import logging

class VerticalTrajFollow():
    def __init__(self):
        rospy.init_node("vertical_trajectory_follow", anonymous=True)
        
        # パラメータ設定
        self.period = rospy.get_param("~period", 40.0)  # 上下運動の周期（秒）
        self.max_height = rospy.get_param("~max_height", 0.2)  # 上に20cm
        self.min_height = rospy.get_param("~min_height", -0.2)  # 下に20cm
        self.nav_rate = rospy.get_param("~nav_rate", 20.0)  # ナビゲーションレート（Hz）

        # 速度計算（period秒で上下運動をするための速度）
        self.velocity_z = (self.max_height - self.min_height) / self.period  # 上下運動の速度
        self.nav_rate = 1 / self.nav_rate  # レートの逆数

        # トピック設定
        self.nav_pub = rospy.Publisher("/gimbalrotor1/uav/nav", FlightNav, queue_size=1)
        self.control_sub = rospy.Subscriber("/gimbalrotor1/debug/pose/pid", PoseControlPid, self.controlCb)

        # 初期設定
        self.flight_nav = FlightNav()
        self.flight_nav.target = FlightNav.COG
        self.flight_nav.pos_xy_nav_mode = FlightNav.POS_VEL_MODE

        self.initial_height = None  # 初期の高さ（tfから取得）

        # TFリスナー
        self.listener = tf.TransformListener()
        
        # 終了シグナルハンドラ設定
        rospy.on_shutdown(self.stopRequest)

        # 初期化待機
        rospy.sleep(0.5)
        self.get_initial_height()  # 初期高さを取得

    def get_initial_height(self):
        """最初の高さをTFから取得"""
        try:
            (trans, rot) = self.listener.lookupTransform("/world", "/gimbalrotor1/lidar_imu", rospy.Time(0))
            self.initial_height = trans[2]  # Z座標（高さ）
            rospy.loginfo(f"Initial height: {self.initial_height}")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF lookup failed, retrying...")
            rospy.sleep(1)
            self.get_initial_height()  # 再試行

    def controlCb(self, msg):
        """制御メッセージの受信（中心座標設定）"""
        rospy.loginfo("制御メッセージを受信しました")
        self.control_sub.unregister()  # 購読解除

    def stopRequest(self):
        """停止処理"""
        rospy.loginfo("上下軌道追従を停止します")
        self.flight_nav.target_vel_x = 0
        self.flight_nav.target_vel_y = 0
        self.flight_nav.target_omega_z = 0
        self.nav_pub.publish(self.flight_nav)

    def main(self):
        """主処理ループ"""
        cnt = 0
        direction = 1  # 1: 上昇, -1: 下降
        while not rospy.is_shutdown() and self.initial_height is not None:
            # 上下方向の変位計算
            displacement_z = self.initial_height + cnt * direction * self.velocity_z * self.nav_rate

            # 高度の設定
            if direction == 1:  # 上昇中
                self.flight_nav.target_pos_z = min(self.max_height + self.initial_height, displacement_z)
            else:  # 下降中
                self.flight_nav.target_pos_z = max(self.min_height + self.initial_height, displacement_z)
            self.flight_nav.target_vel_z = 0.1 * direction
            self.flight_nav.target_vel_x = 0
            self.flight_nav.target_vel_y = 0
            self.flight_nav.target_omega_z = 0

            # メッセージの送信
            self.nav_pub.publish(self.flight_nav)

            # カウンタ更新
            cnt += 1

            # 終了条件
            if cnt * self.nav_rate >= self.period:
                # 上昇と下降を交互に切り替える
                direction *= -1  # 上昇中は下降に、下降中は上昇に
                cnt = 0  # カウントをリセットして、次のサイクルへ

            rospy.sleep(self.nav_rate)

if __name__ == "__main__":
    logging.getLogger("rosout").setLevel(logging.ERROR)
    tracker = VerticalTrajFollow()
    tracker.main()
