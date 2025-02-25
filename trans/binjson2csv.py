import numpy as np
import os
import pandas as pd
import json

def bin_to_csv(binary_path, output_dir):
    """
    将 KITTI 格式的二进制文件转换为CSV，忽略强度维度
    """
    try:
        # 读取二进制文件，包含XYZ和强度
        print(f"正在读取二进制文件: {binary_path}")
        point_cloud = np.fromfile(binary_path, dtype=np.float32).reshape(-1, 4)  # 每个点有4个值：x, y, z, 强度
        
        # 提取XYZ
        xyz = point_cloud[:, :3]
        
        # 创建DataFrame
        df = pd.DataFrame(xyz, columns=['x', 'y', 'z'])

        # 构建输出路径
        filename = os.path.basename(binary_path)
        filename_without_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{filename_without_ext}.csv")
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为CSV
        df.to_csv(output_path, index=False)
        print(f"CSV文件已保存至: {output_path}")
        print(f"点数量: {len(df)}")
        
        return output_path
        
    except Exception as e:
        print(f"转换二进制文件时出错: {str(e)}")
        return None

def load_semantic_mask(json_path):
    """
    从JSON文件加载语义掩码
    """
    try:
        print(f"正在读取JSON文件: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        semantic_mask = np.array(data['pts_semantic_mask'])
        print(f"语义掩码点数量: {len(semantic_mask)}")
        return semantic_mask
        
    except Exception as e:
        print(f"读取JSON文件时出错: {str(e)}")
        return None

def merge_csv_with_mask(csv_path, semantic_mask, output_dir):
    """
    合并CSV特征和语义掩码
    """
    try:
        # 读取CSV
        df = pd.read_csv(csv_path)
        
        if len(df) != len(semantic_mask):
            print(f"警告: 特征数量 ({len(df)}) 与语义掩码数量 ({len(semantic_mask)}) 不匹配")
            return None
            
        # 添加语义掩码
        df['semantic_mask'] = semantic_mask
        
        # 构建输出路径
        filename = os.path.basename(csv_path)
        output_path = os.path.join(output_dir, filename)
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存合并后的CSV
        df.to_csv(output_path, index=False)
        print(f"合并后的CSV文件已保存至: {output_path}")
        
    except Exception as e:
        print(f"合并数据时出错: {str(e)}")

def main():
    # 设定路径
    # binary_path = "data/test/cir/bar_test.bin"
    # json_path = "outputs/preds/bar_test.json"
    # bin_csv_output_dir = "outputs/bincsv"
    # final_output_dir = "outputs/csv"
    binary_path = "data/mySemKitti/sequences/01/velodyne/000001.bin"
    json_path = "outputs/preds/000001.json"
    bin_csv_output_dir = "outputs/bincsv"
    final_output_dir = "outputs/csv"
#测试输入点文件
#     binary_path = "test1.bin"
# #预测的分割推理结果
#     json_path = "outputs/preds/test1.json"
#     #二进制bin转csv
#     bin_csv_output_dir = "outputs/bincsv"
#     final_output_dir = "outputs/csv"
    
    # 第一步：将二进制文件转换为CSV
    csv_path = bin_to_csv(binary_path, bin_csv_output_dir)
    if csv_path is None:
        return
        
    # 第二步：读取语义掩码
    semantic_mask = load_semantic_mask(json_path)
    if semantic_mask is None:
        return
        
    # 第三步：合并数据（如果数量匹配的话）
    merge_csv_with_mask(csv_path, semantic_mask, final_output_dir)

if __name__ == "__main__":
    main()
