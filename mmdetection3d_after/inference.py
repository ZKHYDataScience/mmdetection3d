import argparse
import numpy as np
import torch

# from mmcv import Config
from mmdet3d.apis import init_model, inference_segmentor

def parse_args():
    parser = argparse.ArgumentParser(description='Cylinder3D Inference Example')

    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--out-dir', default='results', help='Directory to save visualization results')
    parser.add_argument('--save-npy', action='store_true', help='Whether to save segmentation result to npy file')
    return parser.parse_args()

def main():
    args = parse_args()
    args.config = "/home/yijian/net/mmdetection3d/configs/cylinder3d/cylinder3d_8xb2-laser-polar-mix-3x_semantickitti.py"
    args.checkpoint = "/home/yijian/net/mmdetection3d/work_dirs/cylinder3d_8xb2-laser-polar-mix-3x_semantickitti/epoch_36.pth"
    args.point_cloud = "/home/yijian/net/mmdetection3d/data/semantickitti/000000.bin"
    # 1. 初始化模型
    model = init_model(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device
    )
    
    # 2. 读取单帧点云
    #    假设你的点云文件是 .bin 格式，内部每个点依次存储 [x, y, z, intensity] (float32)
    #    如果格式不同，需要根据实际情况进行修改
    # pts = np.fromfile(args.point_cloud, dtype=np.float32).reshape(-1, 4)
    pts = args.point_cloud
    
    # 3. 推断（semantic segmentation）
    segmentation_result = inference_segmentor(model, pts)
    # seg_result 通常是一个长度等于点云中点数的一维数组，每个值代表该点的类别索引
    
    # 提取语义分割结果

    semantic_labels = segmentation_result[0].pred_pts_seg
    ptsArray = np.fromfile(args.point_cloud, dtype=np.float32).reshape(-1, 4)
    labelArray = np.array(semantic_labels.cpu().numpy())
    point_cloud_with_labels = np.column_stack((ptsArray, labelArray))  # Add labels to point cloud points

    # 保存结果

    
    # 4. 可视化或保存结果
    # 如果想要可视化，可以使用 mmdet3d 自带的可视化函数 show_seg_result
    # 注意：对于纯点云可视化，通常需要将结果转成可视化工具可读的格式
    # if args.show:
    #     # show_seg_result 会将结果保存到 out_dir 下
    #     # 在命令行中可视化，需要结合Open3D或其它可视化工具
    #     show_seg_result(
    #         data=pts,           # 输入的点云
    #         seg_result=seg_result,
    #         out_dir=args.out_dir,
    #         palette=None,       # 语义分割调色板，可根据需要自定义
    #         show=True           # 仅在本地有 GUI 环境时有效
    #     )
    
    # 可将每个点的类别预测保存为 .npy
    np.save('output.csv', point_cloud_with_labels)
    print(f"Segmentation result with reference labels saved to {args.output}")
if __name__ == '__main__':
    main()