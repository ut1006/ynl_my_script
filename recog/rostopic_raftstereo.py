import sys
sys.path.append('RAFT-Stereo/core')
#python rostopic_demo.py --mixed_precision --corr_implementation reg_cuda --restore_ckpt models/raftstereo-middlebury.pth 



import argparse
import numpy as np
import torch
from tqdm import tqdm
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
import rospy
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
from PIL import Image

from sensor_msgs.msg import PointCloud2,PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pcd2

# Calibration
fx, fy, cx1, cy = 1400.6, 1400.6, 1103.65, 574.575
cx2 = 1102.84
baseline=62.8749 # in millimeters
DEVICE = 'cuda'

# 画像が届くまで保持する変数
left_image = None
right_image = None

# 画像を受信したときのコールバック
def left_image_rect_callback(msg):
    global left_image
    bridge = CvBridge()
    # ROS画像をOpenCVの形式に変換
    left_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def right_image_rect_callback(msg):
    global right_image
    bridge = CvBridge()
    # ROS画像をOpenCVの形式に変換
    right_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def load_image(image_cv):
    img = np.array(image_cv).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def convert_image2(image_tensor):
    # TensorをNumPy配列に変換し、データ型をuint8に設定
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return image_np

def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    bridge = CvBridge()

    while not rospy.is_shutdown():
        if left_image is not None and right_image is not None:
            # Load and pad images
            image1 = load_image(left_image)
            image2 = load_image(right_image)
            image2_unpad = image2
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            # Perform inference
            with torch.no_grad():
                _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
                flow_up = padder.unpad(flow_up).squeeze()
                # print(flow_up.dtype)
                # print(flow_up.size())
#torch.float32
#torch.Size([621, 1104])

                # Convert flow to depth map (negative values represent inverse motion)
                depth_map = -flow_up.cpu().numpy().squeeze() #flow_upがモデル出力であり、コード出力の.npyの中身。
                # print(depth_map.dtype)
                # print(depth_map.shape)
                #float32
#(621, 1104)

                # Normalize depth map to 0-255 for color mapping
                depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

                # Convert normalized depth map to color image for ROS
                depth_colormap = cv2.applyColorMap(depth_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)

                # Show the image for debugging/real-time viewing
                cv2.imshow("Depth Map", depth_colormap)
                cv2.waitKey(1) 
                # print("imshow!")
                '''create and publish point cloud'''
                fields = [  PointField("x",0,PointField.FLOAT32,1),
                                PointField("y",4,PointField.FLOAT32,1),
                                PointField("z",8,PointField.FLOAT32,1),
                                PointField("b",12,PointField.FLOAT32,1),
                                PointField("g",16,PointField.FLOAT32,1),
                                PointField("r",20,PointField.FLOAT32,1)]
                disp = flow_up.cpu().numpy().squeeze()
                depth = (fx * baseline) / (-disp + (cx2 - cx1))
                # print(depth.dtype)
                # print(depth.shape)

 #               torch.float32
#torch.Size([621, 1104])


                H, W = depth.shape
                xx, yy = np.meshgrid(np.arange(W), np.arange(H))
                points_grid = np.stack(((xx-cx1)/fx, (yy-cy)/fy, np.ones_like(xx)), axis=0) * depth/1000
                # print(points_grid.dtype)
                # print(points_grid.shape)
                #float64
#(3, 621, 1104)

                mask = np.ones((H, W), dtype=bool)
                # print(mask.dtype)
                # print(mask.shape)
#bool
#(621, 1104)
                # Remove flying points
                mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
                mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False

                points = points_grid.transpose(1,2,0)[mask]

                rotation_matrix = np.array([
                    [0, -1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]
                ], dtype=np.float32)

                # ポイントを回転
                rotated_points = points @ rotation_matrix.T 



                image2_reshaped = convert_image2(image2_unpad)
                colors = image2_reshaped[mask].astype(np.float32) / 255
                
                data = np.hstack((rotated_points,colors)).astype(np.float32)
                header = Header()
                header.frame_id = "zedm_base_link"
                msg = pcd2.create_cloud(header=header,fields=fields,points=data )                
                pcd_publish.publish(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    # Initialize ROS node
    rospy.init_node('raft_stereo_node')


    # Create ROS publishers and subscribers
    rospy.Subscriber("/zedm/zed_node/left/image_rect_color", ROSImage, left_image_rect_callback)
    rospy.Subscriber("/zedm/zed_node/right/image_rect_color", ROSImage, right_image_rect_callback)
    
    pcd_publish = rospy.Publisher('/test_pointcloud', PointCloud2,queue_size=1)


    args = parser.parse_args()

    # Start demo processing
    demo(args)

    # Close OpenCV windows after process is complete
    cv2.destroyAllWindows()
