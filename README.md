<p align="center">
  <h1 align="center">Skeleton-in-Context: Unified Skeleton Sequence Modeling with In-Context Learning</h1>
  <p align="center">
    CVPR, 2024
    <br />
    <a href="https://github.com/BradleyWang0416/"><strong>Xinshun Wang*</strong></a>
    ·
    <a href="https://github.com/fanglaosi/"><strong>Zhongbin Fang*</strong></a>
    <br />
    <a href="https://xialipku.github.io/"><strong>Xia Li</strong></a>
    ·
    <a href="https://lxtgh.github.io/"><strong>Xiangtai Li</strong></a>
    ·
    <a href="https://www.crcv.ucf.edu/chenchen/"><strong>Chen Chen</strong></a>
    ·
    <a href="https://www.ece.pku.edu.cn/info/1046/2596.htm"><strong>Mengyuan Liu✉</strong></a>
  </p>

  <p align="center">
    <a href='https://arxiv.org/abs/2306.08659'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://fanglaosi.github.io/Point-In-Context_Pages/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
<br />

This is the official PyTorch implementation of the paper "Skeleton-in-Context: Unified Skeleton Sequence Modeling with In-Context Learning" (CVPR 2024).

<div  align="center">    
 <img src="./assets/imgs/Teaser_v2_00.jpg" width = 1000  align=center />
</div>

<!-- ⭐ Our work is the **_first_** to explore in-context learning in 3D point clouds, including task definition, benchmark, and baseline models. -->

# 🙂News
- [Apr 23, 2024] Code is released.
- [Feb 27, 2024] Paper is accepted by CVPR 2024!
- [Dec 07, 2023] Paper is released and GitHub repo is created.

<!-- # ⚡Features

## In-context learning for 3D understanding


- The first work to explore the application of in-context learning in the 3D domain.
- A new framework for tackling multiple tasks (four tasks), which are unified into the same input-output space.
- Can improve the performance of our Point-In-Context (Sep & Cat) by selecting higher-quality prompts.

## New benchmark

- A new multi-task benchmark for evaluating the capability of processing multiple tasks, including reconstruction, denoising, registration, and part segmentation.

## Strong performance

- Surpasses classical models (PointNet, DGCNN, PCT, PointMAE), which are equipped with multi-task heads.
- Surpasses even task-specific models (PointNet, DGCNN, PCT) on registration when given higher-quality prompts. -->

# 😃Run

## 1. Installation
```
conda create -n skeleton_in_context python=3.7 anaconda
conda activate skeleton_in_context
pip install -r requirements.txt
```

## 2. Data Preparation

There are 2 ways to prepare data:

1. You can download ready-to-use data [here]([https://drive.google.com/drive/folders/1NYsgUGdHzWFK_OPwVUm-cl8y-T1Q4MWG](https://drive.google.com/drive/folders/1NYsgUGdHzWFK_OPwVUm-cl8y-T1Q4MWG?usp=sharing)), and unzip the files in ```data/``` (recommended).

2. You can download source data [here] (To be released), and unzip the files in ```data/source_data/```, and pre-process the data yourself by running the following the lines:

```
python data_gen/convert_h36m_PE.py
python data_gen/convert_h36m_FPE.py
python data_gen/convert_amass_MP.py
python data_gen/convert_3dpw_MC.py
python data_gen/calculate_avg_pose.py
```

The final data folder structure should look like this:
```
data/
│
├── 3DPW_MC/
│   ├── train/
│   └── test/
│
├── AMASS/
│   ├── train/
│   └── test/
│
├── H36M/
|   ├── train/
|   └── test/
│
├── H36M_FPE/
|   ├── train/
|   └── test/
|
├── source_data/
|   ├── AMASS/
|   ├── PW3D/
|   └── H36M.pkl
|
└── avg_pose.py
```

## 3. Training
To train Skeleton-in-Context, run the following command:

```
CUDA_VISIBLE_DEVICES=<GPU> python train.py --config configs/default.yaml --checkpoint ckpt/[YOUR_EXP_NAME]
```

## 4. Evaluation
To evaluate Skeleton-in-Context, run the following command:
```
CUDA_VISIBLE_DEVICES=<GPU> python train.py --config configs/default.yaml --evaluate ckpt/[YOUR_EXP_NAME]/[YOUR_CKPT]
```
For example:
```
CUDA_VISIBLE_DEVICES=<GPU> python train.py --config configs/default.yaml --evaluate ckpt/pretrained/latest_epoch.bin
```


<!--# 📚Pretrained Models

Coming soon-->
<!-- | Name                                  | Params | Rec. (CD↓) | Deno. (CD↓) | Reg. (CD↓) | Part Seg. (mIOU↑) |
|---------------------------------------|:------:|:----------:|:----------:|:---------:|:-----------------:|
| [PIC-Sep](https://drive.google.com/file/d/1Dkq5V9LNNGBgxWcPo8tkWC05Yi7DCre3/view?usp=sharing)     | **28.9M**  |  **4.4**   |    **7.5**     |    **8.6**    |     **78.60**     |
| [PIC-Cat](https://drive.google.com/file/d/1Dkq5V9LNNGBgxWcPo8tkWC05Yi7DCre3/view?usp=sharing) | **29.0M**  |  **4.9**   |    **6.0**     |   **14.4**    |     **79.75**     |

> The above results are reimplemented  and are basically consistent with the results reported in the paper. -->

<!-- # ✋Visualization
In-context inference demo (part segmentation, denoising, registration). Our Point-In-Context is designed to perform various tasks on a given query point cloud, adapting its operations based on different prompt pairs. Notably, the PIC has the ability to accurately predict the correct point cloud, even when provided with a clean input point cloud for the denoising task.

![in-context_demo](./assets/gifs/in-context_demo.gif)

Visualization of predictions obtained by our PIC-Sep and their corresponding targets in different point cloud tasks.

![visual](./assets/imgs/visualization_main_00.jpg) -->

# License
MIT License

# Citation
If you find our work useful in your research, please consider citing: 
```
@article{wang2023skeleton,
  title={Skeleton-in-Context: Unified Skeleton Sequence Modeling with In-Context Learning},
  author={Wang, Xinshun and Fang, Zhongbin and Li, Xia and Li, Xiangtai and Chen, Chen and Liu, Mengyuan},
  journal={arXiv preprint arXiv:2312.03703},
  year={2023}
}
```

# Acknowledgement

This work is inspired by [Point-In-Context](https://github.com/fanglaosi/Point-In-Context/). The code
for our work is built upon [MotionBERT](https://github.com/Walter0807/MotionBERT).
Our tribute to these excellent works, and special thanks to the following works: [siMLPe](https://github.com/dulucas/siMLPe), [EqMotion](https://github.com/MediaBrain-SJTU/EqMotion), [STCFormer](https://github.com/zhenhuat/STCFormer), [GLA-GCN](https://github.com/bruceyo/GLA-GCN).
<!-- Thanks to the following excellent works: [PointNet](https://github.com/fxia22/pointnet.pytorch), [DGCNN](https://github.com/WangYueFt/dgcnn), [PCT](https://github.com/MenghaoGuo/PCT), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [ACT](https://github.com/RunpeiDong/ACT), [I2P-MAE](https://github.com/ZrrSkywalker/I2P-MAE), [ReCon](https://github.com/qizekun/ReCon);  -->
