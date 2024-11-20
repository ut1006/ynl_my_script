#!/usr/bin/env python
from __future__ import print_function
from six.moves import input
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import cv2
import os
import csv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from glob import glob
import tf

class MoveGroupPythonInterfaceTutorial(object):
    def __init__(self):
        super(MoveGroupPythonInterfaceTutorial, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface_tutorial", anonymous=True)
        
        # ロボット関連の初期化
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("arm")
        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        # カメラとTFリスナーの設定
        self.left_image_rect = None
        self.right_image_rect = None
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        # ディレクトリ設定
        self.output_dir = "output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # サブスクライバ設定
        rospy.Subscriber("/zedm/zed_node/left/image_rect_color", Image, self.left_image_rect_callback)
        rospy.Subscriber("/zedm/zed_node/right/image_rect_color", Image, self.right_image_rect_callback)

    def left_image_rect_callback(self, msg):
        self.left_image_rect = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def right_image_rect_callback(self, msg):
        self.right_image_rect = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def get_next_filename_count(self, directory):
        files = glob(os.path.join(directory, "*"))
        if not files:
            return 1
        latest_dir = max(files, key=lambda x: int(os.path.basename(x)))
        latest_num = int(os.path.basename(latest_dir))
        return latest_num + 1

    def save_tf_data(self, count, dir_name):
        try:
            (trans, rot) = self.listener.lookupTransform('base_link', 'zedm_left_camera_frame', rospy.Time(0))
            tf_filename = os.path.join(dir_name, f"tf{count:04d}.csv")
            with open(tf_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Translation X', 'Translation Y', 'Translation Z', 'Rotation X', 'Rotation Y', 'Rotation Z', 'Rotation W'])
                writer.writerow([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
            print(f"Saved TF data to {tf_filename}")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Failed to get transform")

    def capture_images_and_save(self):
        if self.left_image_rect is not None and self.right_image_rect is not None:
            count = self.get_next_filename_count(self.output_dir)
            dir_name = os.path.join(self.output_dir, f"{count:04d}")
            os.makedirs(dir_name)

            left_rect_filename = os.path.join(dir_name, f"l{count:04d}.png")
            right_rect_filename = os.path.join(dir_name, f"r{count:04d}.png")

            cv2.imwrite(left_rect_filename, self.left_image_rect)
            cv2.imwrite(right_rect_filename, self.right_image_rect)

            self.save_tf_data(count, dir_name)

            print(f"Saved {left_rect_filename} and {right_rect_filename}")
        else:
            print("No rectified images received yet.")

    def print_current_pose(self):
        current_pose = self.move_group.get_current_pose().pose
        print("Current Position: x={:.3f}, y={:.3f}, z={:.3f}".format(
            current_pose.position.x, 
            current_pose.position.y, 
            current_pose.position.z
        ))
        print("Current Orientation: qx={:.3f}, qy={:.3f}, qz={:.3f}, qw={:.3f}".format(
            current_pose.orientation.x, 
            current_pose.orientation.y, 
            current_pose.orientation.z, 
            current_pose.orientation.w
        ))

    def move_relative(self, dx, dy, dz):
        current_pose = self.move_group.get_current_pose().pose
        new_pose = geometry_msgs.msg.Pose()
        new_pose.position.x = current_pose.position.x + dx
        new_pose.position.y = current_pose.position.y + dy
        new_pose.position.z = current_pose.position.z + dz
        new_pose.orientation = current_pose.orientation

        self.move_group.set_pose_target(new_pose)

        try:
            success = self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            if not success:
                rospy.logerr("Failed to move to the specified relative position.")
            return success
        except Exception as e:
            rospy.logerr("An error occurred: %s", e)
            return False


def main():
    tutorial = MoveGroupPythonInterfaceTutorial()

    while not rospy.is_shutdown():
        tutorial.print_current_pose()

        print("Enter relative movement distance (in cm):")
        try:
            dx = float(input("dx (cm): ")) / 100.0
            dy = float(input("dy (cm): ")) / 100.0
            dz = float(input("dz (cm): ")) / 100.0

            if tutorial.move_relative(dx, dy, dz):
                print("Relative move successful!")
            else:
                print("Failed to complete the relative move.")
                
            # 移動後に撮影するか決める
            capture_input = input("Press 't' to capture images or any other key to skip: ")
            if capture_input.lower() == 't':
                tutorial.capture_images_and_save()

        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except rospy.ROSInterruptException:
            break
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
