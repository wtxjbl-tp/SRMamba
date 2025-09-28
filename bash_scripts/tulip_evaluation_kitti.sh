#!/bin/bash

args=(
    --eval
    --mc_drop
    --noise_threshold 0.03
    --model_select tulip_base
    --pixel_shuffle
    --circular_padding
    --patch_unmerging
    --log_transform
    # Dataset
    --dataset_select kitti
    --data_path_low_res /DATA/TULIP/kitti_raw_data/
    --data_path_high_res /DATA/TULIP/kitti_raw_data/
    --save_pcd
    # WandB Parameters
    --run_name init_TULIP
    --entity chencuit28-peak
#   --wandb_disabled
    --project_name kitti_evaluation
    --output_dir ./trained/tulip_kitti.pth
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

#torchrun --nproc_per_node=1 SRMamba/main_lidar_upsampling.py "${args[@]}"
python tulip/main_lidar_upsampling.py "${args[@]}"