# MMDetection3D  配置说明项目（Cylinder3D）

## 项目简介

本项目主要包含了 MMDetection3D 框架下 Cylinder3D 模型的配置文件，提供了原始版本和修改版本。

## 文件结构

```
.
├── mmdetection3d_original/    # 原始 MMDetection 的 Cylinder3D 模型配置
│   ├── [配置文件列表]
│   └── ...
├── mmdetection3d_after/      # 修改后的 MMDetection 的 Cylinder3D 模型配置
│   ├── [配置文件列表]
│   └── ...
└── 安装教程.txt              # 详细的环境配置和安装步骤说明
```

## 配置文件说明

### 原始配置 (mmdetection3d_original)
- 包含原始 Cylinder3D 模型在 MMDetection3D 框架下的标准配置
- 保持了原始论文中的模型结构和参数设置

### 修改后配置 (mmdetection3d_after)
- 包含经过对应自身数据修改的 Cylinder3D 模型配置


## 安装教程

1. 创建 Conda 环境
   conda create --name openmmlab python=3.8 -y

2. 激活环境
   conda activate openmmlab

3. 安装 PyTorch 和 torchvision
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

4. 安装 OpenMIM 和 MM 工具包
   pip install openmim
   mim install mmengine
   mim install mmdet
   mim install mmsegmentation
   mim install mmdet3d

5. 安装 MMCV
   从以下链接下载适合环境的 .whl 文件：
   https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

   cd 到下载目录
   pip install mmcv-2.1.0-cp38-cp38-manylinux1_x86_64.whl

6. 测试安装
   # 下载 mmdet3d 配置文件
   mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .

   # 运行测试
   python demo/pcd_demo.py demo/data/kitti/000008.bin temp/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py temp/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show