#!/bin/bash

topics=$(rostopic list |grep "^/gimbalrotor1/")
rosbag record \
    /tf \
    /tf_static \
    /gimbalrotor1/Odometry_precede \
    /gimbalrotor1/imu \
    /gimbalrotor1/livox/lidar \
    /zedm/zed_node/imu/data \
    /zedm/zed_node/left/image_rect_color \
    /zedm/zed_node/right/image_rect_color \
    
    -o /media/kamadagpu/JetsonSSD/rosbag_1124_imu_test/

    # lidar点群を取るとimageがブレる

    # /gimbalrotor1/cloud_registered \

    # /gimbalrotor1/livox/imu \