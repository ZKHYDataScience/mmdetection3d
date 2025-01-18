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

## 安装步骤

详细的安装步骤请参考 `安装教程.txt`，主要包括：

1. Conda 环境配置
2. PyTorch 安装
3. MM 系列工具包安装
4. MMCV 安装
5. 环境测试

## 注意事项

- 确保已正确安装所有依赖包
- 训练前检查 `my_cylinder3d.py` 配置文件中的参数设置
- 预测时确保输入点云文件格式正确
- 如遇到 CUDA 相关错误，请检查 GPU 驱动和 CUDA 版本是否匹配

