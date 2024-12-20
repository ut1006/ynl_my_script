#!/usr/bin/env python

import sys
import time
import rospy
import math
import signal
from aerial_robot_msgs.msg import FlightNav, PoseControlPid

class CircTrajFollow():
    def __init__(self):
        # パラメータ設定
        self.period = rospy.get_param("~period", 40.0)
        self.radius = rospy.get_param("~radius", 0.5)
        self.init_theta = rospy.get_param("~init_theta", 0.0)
        self.yaw = rospy.get_param("~yaw", True)
        self.loop = rospy.get_param("~loop", False)
        self.nav_rate = rospy.get_param("~nav_rate", 20.0)  # Hz

        # 速度と角速度を計算
        self.omega = 2 * math.pi / self.period
        self.velocity = self.omega * self.radius
        self.nav_rate = 1 / self.nav_rate

        # トピック設定
        self.nav_pub = rospy.Publisher("/gimbalrotor1/uav/nav", FlightNav, queue_size=1)
        self.control_sub = rospy.Subscriber("/gimbalrotor1/debug/pose/pid", PoseControlPid, self.controlCb)

        # 中心座標の初期化
        self.center_pos_x = None
        self.center_pos_y = None

        # FlightNav メッセージ初期化
        self.flight_nav = FlightNav()
        self.flight_nav.target = FlightNav.COG
        self.flight_nav.pos_xy_nav_mode = FlightNav.POS_VEL_MODE
        if self.yaw:
            self.flight_nav.yaw_nav_mode = FlightNav.POS_VEL_MODE

        # 終了シグナルハンドラ設定
        signal.signal(signal.SIGINT, self.stopRequest)

        # 初期化待機
        rospy.sleep(0.5)

    def controlCb(self, msg):
        """中心位置を算出"""
        self.initial_target_yaw = msg.yaw.target_p
        self.center_pos_x = msg.x.target_p - math.cos(self.init_theta) * self.radius
        self.center_pos_y = msg.y.target_p - math.sin(self.init_theta) * self.radius

        rospy.loginfo("中心座標が設定されました: [%f, %f]", self.center_pos_x, self.center_pos_y)
        self.control_sub.unregister()  # 購読を解除

    def stopRequest(self, signal, frame):
        """停止処理"""
        rospy.loginfo("軌道追従を停止します")
        self.flight_nav.target_vel_x = 0
        self.flight_nav.target_vel_y = 0
        self.flight_nav.target_omega_z = 0
        self.nav_pub.publish(self.flight_nav)
        sys.exit(0)

    def main(self):
        """主処理ループ"""
        cnt = 0

        while not rospy.is_shutdown():
            # 中心座標が設定されるまで待機
            if self.center_pos_x is None:
                rospy.loginfo_throttle(1.0, "制御メッセージを受信待ち")
                rospy.sleep(self.nav_rate)
                continue

            # 軌道追従計算
            theta = self.init_theta + cnt * self.nav_rate * self.omega
            self.flight_nav.target_pos_x = self.center_pos_x + math.cos(theta) * self.radius
            self.flight_nav.target_pos_y = self.center_pos_y + math.sin(theta) * self.radius
            self.flight_nav.target_vel_x = -math.sin(theta) * self.velocity
            self.flight_nav.target_vel_y = math.cos(theta) * self.velocity

            if self.yaw:
                self.flight_nav.target_yaw = self.initial_target_yaw + cnt * self.nav_rate * self.omega
                self.flight_nav.target_omega_z = self.omega

            # メッセージの送信
            self.nav_pub.publish(self.flight_nav)

            # カウンタ更新
            cnt += 1

            # 軌道終了条件
            if cnt == int(self.period / self.nav_rate):
                if self.loop:
                    cnt = 0
                else:
                    rospy.sleep(0.1)
                    self.stopRequest(None, None)

            rospy.sleep(self.nav_rate)

if __name__ == "__main__":
    rospy.init_node("circle_trajectory_follow", anonymous=True)
    tracker = CircTrajFollow()
    tracker.main()
