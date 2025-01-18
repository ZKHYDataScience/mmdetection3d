
#训练
python tools/train.py my_cylinder3d.py 

#预测
python demo/pcd_seg_demo.py test1.bin my_cylinder3d.py work_dirs/my_cylinder3d/epoch_128.pth --show
