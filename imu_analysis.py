#!/usr/bin/env python

import rospy
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def extract_imu_data(bagfile, topic):
    """ROSbagからIMUデータを抽出"""
    timestamps = []
    acc_data = []
    gyro_data = []
    mag_data = []
    angles = []

    with rosbag.Bag(bagfile, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            timestamps.append(t.to_sec())
            acc_data.append([msg.acc_data[0], msg.acc_data[1], msg.acc_data[2]])
            gyro_data.append([msg.gyro_data[0], msg.gyro_data[1], msg.gyro_data[2]])
            mag_data.append([msg.mag_data[0], msg.mag_data[1], msg.mag_data[2]])
            angles.append([msg.angles[0], msg.angles[1], msg.angles[2]])

    return np.array(timestamps), np.array(acc_data), np.array(gyro_data), np.array(mag_data), np.array(angles)

def plot_time_series(timestamps, data, labels, title):
    """時系列データのプロット"""
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.plot(timestamps, data[:, i], label=label)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

def plot_frequency(data, sample_rate, labels, title):
    """周波数特性のプロット"""
    n = len(data)
    freq = fftfreq(n, d=1/sample_rate)
    fft_values = fft(data, axis=0)

    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.plot(freq[:n // 2], np.abs(fft_values[:n // 2, i]), label=label)
    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    rospy.init_node("imu_data_visualization")
    
    # ROSbagファイルとトピックを指定
    bagfile = "/home/kamada/2024-11-19-09-30-29-stable-flight-morning-height-1p3.bag"
    # topic = "/gimbalrotor1/imu"
    topic = "/gimbalrotor1/livox/imu"
    
    # データ取得
    timestamps, acc_data, gyro_data, mag_data, angles = extract_imu_data(bagfile, topic)
    
    # サンプリングレートの計算
    sample_rate = 1 / np.mean(np.diff(timestamps))
    
    # 可視化：時系列データ
    plot_time_series(timestamps, acc_data, ["Ax", "Ay", "Az"], "Acceleration")
    plot_time_series(timestamps, gyro_data, ["Gx", "Gy", "Gz"], "Gyroscope")
    plot_time_series(timestamps, mag_data, ["Mx", "My", "Mz"], "Magnetic Field")
    plot_time_series(timestamps, angles, ["Roll", "Pitch", "Yaw"], "Angles")
    
    # 可視化：周波数特性
    plot_frequency(acc_data, sample_rate, ["Ax", "Ay", "Az"], "Frequency Spectrum of Acceleration")
