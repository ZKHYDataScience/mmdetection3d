# MMDetection3D 配置说明项目（Cylinder3D）

## 项目简介

本项目主要包含了基于 MMDetection3D 框架的 Cylinder3D 模型配置文件，针对特定数据进行了修改。

## 文件结构

```
.
├── tools/                     # 训练相关工具
│   └── train.py              # 训练脚本
├── demo/                     # 演示和测试工具
│   └── pcd_seg_demo.py      # 点云分割演示脚本
├── work_dirs/               # 训练输出目录
│   └── my_cylinder3d/       # 模型训练结果
│       └── epoch_128.pth    # 训练好的模型权重
├── my_cylinder3d.py         # 模型配置文件
└── 安装教程.txt             # 详细的环境配置和安装步骤说明
```

## 使用说明

### 训练模型

使用以下命令进行模型训练：

```bash
python tools/train.py my_cylinder3d.py
```

训练完成后，模型权重文件将保存在 `work_dirs/my_cylinder3d/` 目录下。

### 预测/推理

使用以下命令进行点云分割预测：

```bash
python demo/pcd_seg_demo.py test1.bin my_cylinder3d.py work_dirs/my_cylinder3d/epoch_128.pth --show
```

参数说明：
- `test1.bin`: 输入的点云文件（替补为需要预测的点云数据）
- `my_cylinder3d.py`: 模型配置文件
- `work_dirs/my_cylinder3d/epoch_128.pth`: 训练好的模型权重文件（替换位训练权重）
- `--show`: 显示预测结果

## 环境要求

- Python 3.8
- CUDA 11.8
- PyTorch 2.0.1
- MMCV 2.1.0
- MMDetection3D 及相关依赖

## 安装教程


驱动（cpu可跳过）

查看：nvidia-smi

如果有返回相关驱动信息，说明驱动有安装，直接跳过驱动这块，直接后续安装

如果没有，则安装驱动

1.查找驱动：apt search nvidia

2.会显示合适的驱动版本，比如安装其中的535：sudo apt install nvidia-driver-535

3.赋予权限：chmod 755 NVIDIA-Linux-x86_64-535.216.01.run

4.安gcc：sudo apt update

sudo apt install gcc g++

5.安内核：sudo apt install linux-headers-$(uname -r)

6.升级内核：sudo update kernel

7.进入虚拟终端：Ctrl + Alt + f1 

8. 停止显示管理：systemctl stop lightdm
   
9.安装：sudo bash ./NVIDIA-Linux-x86_64-535.216.01.run

10.重启系统：sudo reboot

安装cuda（cpu可跳过）

1. wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

2. sudo sh cuda_11.8.0_520.61.05_linux.run

3.设置环境变量

 nano ~/.bashrc

export CUDA_HOME=/usr/local/cuda

export PATH=$CUDA_HOME/bin:$PATH

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

Source ~/.bashrc

安装anaconda

1. 安装Linux的Anaconda安装包

2. 可将安装包移动到Linux的/home/username/下，或其他有权限能读写的路径

3. 安装Anaconda：bash Anaconda3-2024.10-1-Linux-x86_64.sh

（或快速安装：sudo bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p /home/lenovo/anaconda3）

4. 导入路径：source ~/.bashrc

环境

1. 创建 Conda 环境

conda create --name openmmlab python=3.8 -y

2. 激活环境

conda activate openmmlab

3. 安装 PyTorch 和 torchvision（cpu版可替换对应:pip install torch==2.4.1  torchvision==0.19.1）

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

4. 安装 OpenMIM 和 MM 工具包
pip install openmim
mim install mmengine
mim install mmdet
mim install mmsegmentation
mim install mmdet3d

5. 安装 MMCV
（cpu版替换，网址对应：https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html 下载选择，选对应系统版本，以wins举例：mmcv-2.1.0-cp38-cp38-win_amd64.whl ；cd 到此路径，安装mmcv
pip install mmcv-2.1.0-cp38-cp38-win_amd64.whl）
直接从 OpenMMLab 的源头下载并安装 mmcv。访问以下链接获取适合环境的 .whl 文件：
https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

cd 到此路径，安装mmcv
pip install mmcv-2.1.0-cp38-cp38-manylinux1_x86_64.whl

6.安装torchsparse
pip install torchsparse==1.4.0

7. 测试是否安装成功
下载 mmdet3d 配置文件
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .

测试，在下载配置文件和模型后，运行测试，验证模型推理功能。
python demo/pcd_demo.py demo/data/kitti/000008.bin temp/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py temp/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show


## 注意事项

- 确保已正确安装所有依赖包
- 训练前检查 `my_cylinder3d.py` 配置文件中的参数设置
- 预测时确保输入点云文件格式正确
- 如遇到 CUDA 相关错误，请检查 GPU 驱动和 CUDA 版本是否匹配

