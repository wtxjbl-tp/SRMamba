#!/bin/bash


args=(
    --batch_size 16
    --epochs 600
    --num_workers 2
    --lr 5e-4
    --weight_decay 0.01
    --warmup_epochs 60
    # Model parameters
    --model_select SRMamba_tiny
    --pixel_shuffle # improve
    --circular_padding # improve
    --log_transform # improve
    --patch_unmerging # improve
    # Dataset
    --dataset_select nuScenes
    --data_path_low_res /path_to_kitti/
    --data_path_high_res /path_to_kitti/
    # WandB Parameters
    --run_name SRMamba_tiny
    --entity myentity
     --wandb_disabled
    --project_name experiment_kitti
    --output_dir ./experiment/kitti/SRMamba_tiny
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

# real batch size in training = batch_size * nproc_per_node
torchrun --nproc_per_node=4 SRMamba/main_lidar_upsampling.py "${args[@]}"
