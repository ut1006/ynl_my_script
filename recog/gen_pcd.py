import argparse
import numpy as np
from pathlib import Path
from imageio.v3 import imread
from datetime import datetime
import glob
import re
import open3d as o3d
# Calibration parameters
fx, fy, cx1, cy = 1400.6, 1400.6, 1103.65, 574.575
cx2 = 1102.84
baseline = 62.8749  # in millimeters
# 回転行列 (Z軸周りに-90度)
Rotation_Z_90 = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1]
])

import struct

def write_ply(filename, points, colors):
    header = f'''ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    with open(filename, 'wb') as f:
        # Write the header
        f.write(header.encode('utf-8'))
        
        # Write each point and color as binary data
        for point, color in zip(points, colors):
            f.write(struct.pack('<fffBBB',
                                point[0], point[1], point[2],
                                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))

def process_pair(disp_path, image_path):
    output_dir = Path(image_path).parent
    parent_dir_name = output_dir.name
    ply_filename = output_dir / f'{parent_dir_name}.ply'

    # # Skip processing if PLY file already exists
    # if ply_filename.exists():
    #     print(f"Skipping {output_dir} as PLY file already exists.")
    #     return

    # Load disparity and image
    disp = np.load(disp_path)
    image = imread(image_path)

    # Inverse project. PLY origin is on the left camera and x↑, y↓, z×. mm!!
    depth = (fx * baseline) / (-disp + (cx2 - cx1))
    H, W = depth.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
  
    # points_grid = np.stack(((xx - cx1) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=0) * depth 単位はメートルに
    points_grid = np.stack(((xx-cx1/2) / fx, (yy-cy/2) / fy, np.ones_like(xx)), axis=0) * depth /1000
    mask = np.ones((H, W), dtype=bool)

    # Remove flying points
    mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
    mask[:, 1:][np.abs(depth[:, 1:] - depth[:, :-1]) > 1] = False

    points = points_grid.transpose(1, 2, 0)[mask]
    colors = image[mask].astype(np.float64) / 255

    # Z軸中心で90度回転を適用
    points_rotated = np.dot(points, Rotation_Z_90.T)

    # Get the directory of the input image to save the PLY file there
    output_dir = Path(image_path).parent
    parent_dir_name = output_dir.name
    ply_filename = output_dir / f'{parent_dir_name}.ply'

    # Save the rotated PLY file
    write_ply(ply_filename, points_rotated, colors)
    print(f'Saved PLY file to: {ply_filename}')
def natural_sort_key(s):
    """Sort strings in natural order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def main(disp_pattern, image_pattern):
    # Find all matching disparity and image files
    disp_files = sorted(glob.glob(disp_pattern), key=natural_sort_key)
    image_files = sorted(glob.glob(image_pattern), key=natural_sort_key)

    # Ensure both lists are the same length
    if len(disp_files) != len(image_files):
        print(f"Warning: Number of disparity files ({len(disp_files)}) does not match number of image files ({len(image_files)}).")

    # Process each pair
    for disp_file, image_file in zip(disp_files, image_files):
        process_pair(disp_file, image_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disp', required=True, help='Glob pattern for disparity numpy files (.npy)')
    parser.add_argument('--image', required=True, help='Glob pattern for image files')
    
    args = parser.parse_args()
    
    main(args.disp, args.image)
