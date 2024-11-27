#!/bin/bash
#Run this in my_moveit directory.

cd RAFT-Stereo

python my_demo.py --restore_ckpt models/raftstereo-middlebury.pth \
--mixed_precision -l=../images/*/l*.png \
-r=../images/*/r*.png --save_numpy \
--corr_implementation reg_cuda\


cd ..

python gen_pcd.py --disp=images/*/*.npy --image=images/*/l*.png 

python tf_ply.py