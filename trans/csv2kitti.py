import os
import numpy as np
import pandas as pd
import pickle


def convert_csv_to_kitti(csv_file_path, bin_file_path, label_file_path):
    # 读取CSV文件
    data = pd.read_csv(csv_file_path, header=None)

    # 提取xyz和label
    xyz = data.iloc[:, :3].values
    labels = data.iloc[:, 3].values

    # 处理标签中的NaN值，将其替换为0或其他合适的值
    labels = np.nan_to_num(labels, nan=0).astype(np.uint32)

    # 添加强度值
    intensity = np.full((xyz.shape[0], 1), 0.9)
    point_cloud = np.hstack((xyz, intensity))

    # 保存为bin文件
    point_cloud.astype(np.float32).tofile(bin_file_path)

    # 保存为label文件
    labels.tofile(label_file_path)

    return labels


def process_directory(input_directory, output_directory):
    # 创建输出目录结构
    sequence_00_dir = os.path.join(output_directory, "sequences", "00", "velodyne")
    sequence_01_dir = os.path.join(output_directory, "sequences", "01", "velodyne")
    label_00_dir = os.path.join(output_directory, "sequences", "00", "labels")
    label_01_dir = os.path.join(output_directory, "sequences", "01", "labels")

    os.makedirs(sequence_00_dir, exist_ok=True)
    os.makedirs(sequence_01_dir, exist_ok=True)
    os.makedirs(label_00_dir, exist_ok=True)
    os.makedirs(label_01_dir, exist_ok=True)

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]

    train_infos = []
    val_infos = []
    all_labels = []

    # 处理前两个CSV文件到01目录(验证集)
    for i, csv_file in enumerate(csv_files[:2]):
        relative_bin_path = os.path.join("sequences", "01", "velodyne", f"{i:06d}.bin")
        relative_label_path = os.path.join("sequences", "01", "labels", f"{i:06d}.label")

        # 用于实际保存文件的完整路径
        bin_file_path = os.path.join(output_directory, relative_bin_path)
        label_file_path = os.path.join(output_directory, relative_label_path)

        # 转换文件并获取标签
        labels = convert_csv_to_kitti(
            os.path.join(input_directory, csv_file),
            bin_file_path,
            label_file_path
        )
        all_labels.extend(labels[~np.isnan(labels)])  # 只保存非NaN的标签值

        info = {
            'lidar_points': {
                'lidar_path': relative_bin_path.replace("\\", "/"),
                'num_pts_feats': 4
            },
            'pts_semantic_mask_path': relative_label_path.replace("\\", "/"),
            'sample_idx': f"{i:06d}"
        }
        val_infos.append(info)

    # 处理剩余的CSV文件到00目录(训练集)
    for i, csv_file in enumerate(csv_files[2:]):
        relative_bin_path = os.path.join("sequences", "00", "velodyne", f"{i:06d}.bin")
        relative_label_path = os.path.join("sequences", "00", "labels", f"{i:06d}.label")

        # 用于实际保存文件的完整路径
        bin_file_path = os.path.join(output_directory, relative_bin_path)
        label_file_path = os.path.join(output_directory, relative_label_path)

        # 转换文件并获取标签
        labels = convert_csv_to_kitti(
            os.path.join(input_directory, csv_file),
            bin_file_path,
            label_file_path
        )
        all_labels.extend(labels[~np.isnan(labels)])  # 只保存非NaN的标签值

        info = {
            'lidar_points': {
                'lidar_path': relative_bin_path.replace("\\", "/"),
                'num_pts_feats': 4
            },
            'pts_semantic_mask_path': relative_label_path.replace("\\", "/"),
            'sample_idx': f"{i:06d}"
        }
        train_infos.append(info)

    # 计算最大标签值，确保没有NaN
    all_labels = np.array(all_labels)
    all_labels = all_labels[~np.isnan(all_labels)]  # 过滤掉NaN值
    max_label = int(np.max(all_labels)) if len(all_labels) > 0 else 0

    # 准备保存的数据
    train_data = {
        'data_list': train_infos,
        'metainfo': {
            'dataset': 'semantickitti',
            'seg_label_mapping': None,
            'max_label': max_label
        }
    }

    val_data = {
        'data_list': val_infos,
        'metainfo': {
            'dataset': 'semantickitti',
            'seg_label_mapping': None,
            'max_label': max_label
        }
    }

    # 保存pkl文件
    with open(os.path.join(output_directory, 'mySemKitti_infos_train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)

    with open(os.path.join(output_directory, 'mySemKitti_infos_val.pkl'), 'wb') as f:
        pickle.dump(val_data, f)

    with open(os.path.join(output_directory, 'mySemKitti_infos_test.pkl'), 'wb') as f:
        pickle.dump(val_data, f)


# 指定输入和输出目录
input_directory = "D:/company/data/circle1000"
output_directory = "D:/company/data/circle_kitti1000"

process_directory(input_directory, output_directory)