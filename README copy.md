# MMDetection3D 配置说明项目（spvcnn）

## 项目简介

本项目主要包含了基于 MMDetection3D 框架的 spvcnn 模型配置文件，针对特定数据进行了修改。

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
- `test1.bin`: 输入的点云文件（替换为需要预测的点云数据）
- `my_cylinder3d.py`: 模型配置文件
- `work_dirs/my_cylinder3d/epoch_128.pth`: 训练好的模型权重文件
- `--show`: 显示预测结果

## 环境要求

- Python 3.8
- CUDA 11.8
- PyTorch 2.0.1
- MMCV 2.1.0
- MMDetection3D 及相关依赖

## 安装教程

### 安装驱动（CPU 可跳过）

1. 查看驱动状态：
   ```bash
   nvidia-smi
   ```
   如有返回驱动信息，则跳过驱动安装

2. 安装驱动步骤：
   ```bash
   apt search nvidia
   sudo apt install nvidia-driver-535
   chmod 755 NVIDIA-Linux-x86_64-535.216.01.run
   sudo apt update
   sudo apt install gcc g++
   sudo apt install linux-headers-$(uname -r)
   sudo update kernel
   ```

3. 进入虚拟终端并完成安装：
   ```bash
   # Ctrl + Alt + F1 进入虚拟终端
   systemctl stop lightdm
   sudo bash ./NVIDIA-Linux-x86_64-535.216.01.run
   sudo reboot
   ```

### 安装 CUDA（CPU 可跳过）

1. 下载并安装 CUDA：
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run
   ```

2. 配置环境变量：
   ```bash
   nano ~/.bashrc
   ```
   添加以下内容：
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```
   使环境变量生效：
   ```bash
   source ~/.bashrc
   ```

### 安装 Anaconda

1. 下载 Linux 版 Anaconda 安装包
2. 移动安装包到合适路径
3. 安装：
   ```bash
   bash Anaconda3-2024.10-1-Linux-x86_64.sh
   # 或快速安装：
   sudo bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p /home/lenovo/anaconda3
   ```
4. 更新环境变量：
   ```bash
   source ~/.bashrc
   ```

### 配置环境

1. 创建并激活 Conda 环境：
   ```bash
   conda create --name openmmlab python=3.8 -y
   conda activate openmmlab
   ```

2. 安装 PyTorch 和 torchvision：
   ```bash
   # GPU 版本
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   # CPU 版本
   # pip install torch==2.4.1 torchvision==0.19.1
   ```

3. 安装 OpenMIM 和 MM 工具包：
   ```bash
   pip install openmim
   mim install mmengine
   mim install mmdet
   mim install mmsegmentation
   mim install mmdet3d
   ```

4. 安装 MMCV：
   ```bash
   # GPU 版本（根据实际环境选择对应版本）
   # 从 https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html 下载
   pip install mmcv-2.1.0-cp38-cp38-manylinux1_x86_64.whl
   
   # CPU 版本
   # 从 https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html 下载
   # pip install mmcv-2.1.0-cp38-cp38-win_amd64.whl
   ```

5. 安装 torchsparse：
   ```bash
   pip install torchsparse==1.4.0
   ```

### 测试安装

1. 下载测试配置文件：
   ```bash
   mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .
   ```

2. 运行测试：
   ```bash
   python demo/pcd_demo.py demo/data/kitti/000008.bin temp/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py temp/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show
   ```

## 注意事项

- 确保已正确安装所有依赖包
- 训练前检查 `my_cylinder3d.py` 配置文件中的参数设置
- 预测时确保输入点云文件格式正确
- 如遇到 CUDA 相关错误，请检查 GPU 驱动和 CUDA 版本是否匹配
