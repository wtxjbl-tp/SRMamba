#!/bin/bash

args=(
    --eval
    --mc_drop
    --noise_threshold 0.03
    --model_select SRMamba_tiny
    --pixel_shuffle
    --circular_padding
    --patch_unmerging
    --log_transform
    # Dataset
    --dataset_select kitti
    --data_path_low_res /kitti_raw_data/
    --data_path_high_res /kitti_raw_data/
    --save_pcd
    # WandB Parameters
    --run_name SRMamba
    --entity entity_name
#   --wandb_disabled
    --project_name kitti_evaluation
    --output_dir ./trained/SRMamba_kitti.pth
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

torchrun --nproc_per_node=4 SRMamba/main_lidar_upsampling.py "${args[@]}"
