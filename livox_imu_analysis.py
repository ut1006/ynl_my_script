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

    with rosbag.Bag(bagfile, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            timestamps.append(t.to_sec())
            acc_data.append([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            gyro_data.append([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])

    return np.array(timestamps), np.array(acc_data), np.array(gyro_data)

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

def plot_frequency(data, sample_rate, labels, title, freq_limit=1000):
    """正の周波数成分のプロット (周波数制限あり)"""
    n = len(data)
    freq = fftfreq(n, d=1/sample_rate)  # 周波数軸の計算
    fft_values = fft(data, axis=0)  # FFTを実行

    # 正の周波数成分のみ取り出し
    pos_freq_idx = np.where(freq > 0)  # 正の周波数のインデックス
    freq_pos = freq[pos_freq_idx]  # 正の周波数
    fft_values_pos = fft_values[pos_freq_idx]  # 正の周波数成分のFFT値

    # 周波数制限を設定
    max_freq_idx = np.where(freq_pos <= freq_limit)  # 指定した周波数以下のインデックス
    freq_pos_limited = freq_pos[max_freq_idx]  # 制限した周波数
    fft_values_pos_limited = fft_values_pos[max_freq_idx]  # 制限した周波数成分

    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.plot(freq_pos_limited, np.abs(fft_values_pos_limited[:, i]), label=label)
    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    rospy.init_node("imu_data_visualization")
    
    # ROSbagファイルとトピックを指定
    bagfile = "/media/kamadagpu/JetsonSSD/rosbag_1124_imu_test/\
_2024-11-24-13-14-04_land_fail.bag\
"
    topic = "/zedm/zed_node/imu/data"
    
    # データ取得
    timestamps, acc_data, gyro_data = extract_imu_data(bagfile, topic)
    
    # サンプリングレートの計算
    sample_rate = 1 / np.mean(np.diff(timestamps))
    
    # 可視化：時系列データ
    plot_time_series(timestamps, acc_data, ["Ax", "Ay", "Az"], "Acceleration")
    plot_time_series(timestamps, gyro_data, ["Gx", "Gy", "Gz"], "Gyroscope")
    
    # 可視化：周波数特性
    plot_frequency(acc_data, sample_rate, ["Ax", "Ay", "Az"], "Frequency Spectrum of Acceleration", freq_limit=1000)
