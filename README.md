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
bash bash_scripts/create_kitti_dataset.sh
```
## Train
```
bash bash_scripts/SRMamba_upsampling_kitti.sh
```
## Val
```
bash bash_scripts/SRMamba_evaluation_kitti.sh
```
