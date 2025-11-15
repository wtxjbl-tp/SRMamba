## Installation
### clone code
```
git clone https://github.com/wtxjbl-tp/SRMamba.git
cd SRMamba
```

### create environment
```
conda env create -f environment.yml
conda activate SRMamba
```
## Data Preparation
We use [kitti-360](https://www.cvlibs.net/datasets/kitti/index.php) and [nuScenes](https://www.nuscenes.org/) as our datasets, and you can download the datasets from their open-source websites.
### range image
Projecting point clouds during training severely affects efficiency, so the point clouds can be pre-converted to range images using the following script:
```
python sample_kitti_dataset.py --num_data_train 20000 --num_data_val 2500 --output_path_name_train "train_save_path" --output_path_name_val "val_save_path" --input_path "input_path" --create_val
```
## Train
```
bash main_lidar_upsampling.py bash_scrips/SRMamba_upsampling_kitti.sh
```
## Val
```
bash main_lidar_upsampling.py bash_scrips/SRMamba_evaluation_kitti.sh
```
