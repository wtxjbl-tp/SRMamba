#!/bin/bash

args=(
    --num_data_train 20000
    --num_data_val 2500
    --output_path_name_train train
    --output_path_name_val val
    --input_path ./kitti_raw_data/
    --create_val
    )

python kitti_utils/sample_kitti_dataset.py "${args[@]}"
