#!/usr/bin/env python

import sys
import rospy
from aerial_robot_msgs.msg import FlightNav
import signal

class ZDirectionPositionVelocityControl:
    def __init__(self):
        # パラメータ
        self.velocity_z = rospy.get_param("~velocity_z", 0.10)  # Z方向の速度 [m/s]
        self.target_position_z = rospy.get_param("~target_position_z", 0.50)  # Z方向の目標位置 [m]
        self.nav_rate = rospy.get_param("~nav_rate", 20.0)  # コマンドの送信周期 [Hz]

        # Publisher設定
        self.nav_pub = rospy.Publisher("/gimbalrotor1/uav/nav", FlightNav, queue_size=1)

        # メッセージ初期化
        self.flight_nav = FlightNav()
        self.flight_nav.target = FlightNav.COG  # 操作対象をCOGに設定
        self.flight_nav.pos_z_nav_mode = FlightNav.POS_VEL_MODE  # Z方向の制御モードを位置・速度モードに設定
        self.flight_nav.target_vel_z = self.velocity_z  # Z方向の目標速度を設定
        self.flight_nav.target_pos_z = self.target_position_z  # Z方向の目標位置を設定

        # 終了処理を設定
        signal.signal(signal.SIGINT, self.stopRequest)

        rospy.sleep(0.5)  # 初期化待ち

    def stopRequest(self, signal, frame):
        """安全に停止する"""
        rospy.loginfo("Z方向制御を停止します")
        self.flight_nav.target_vel_z = 0  # 停止指令
        self.nav_pub.publish(self.flight_nav)  # 最後の停止指令を送信
        sys.exit(0)

    def main(self):
        """位置・速度指令を定期送信するメイン処理"""
        rate = rospy.Rate(self.nav_rate)  # nav_rateの周期でループ

        while not rospy.is_shutdown():
            # FlightNavメッセージを送信
            self.nav_pub.publish(self.flight_nav)
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("z_direction_position_velocity_control", anonymous=True)
    controller = ZDirectionPositionVelocityControl()
    controller.main()
