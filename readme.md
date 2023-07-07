# 细胞分割网络LS-Trans-Unet
## 文件列表
- data/ ——数据集（位于服务器）
- results/ ——网络权值及可视化结果（位于服务器）
- networks/ ——网络模型
  - unet/ ——U-Net
  - vision_transformer/ ——ViT
  - ming_net.py ——LS-Trans-Unet
  - trans_unet.py ——Trans-Unet
- utils/ ——工具包
  - data_loading.py ——数据加载
  - data_pre.py ——数据预处理
  - dice_score.py ——dice指标
  - loss.py ——损失函数
  - utils.py ——其他工具函数
- train.py ——训练
- predict.py ——预测
- evaluate.py ——评估
- requirements.txt ——python环境需求
- readme.md ——工程说明
- view.ipynb ——jupyter调试脚本
> 文件内容详细介绍见《成果验收说明书》
---
## 使用说明
- 环境配置
```commandline
pip install -r requirements.txt
```
- 训练
```commandline
python train.py --epochs 20 --batch-size 8 --learning-rate 1e-5 --scale 1.0 --channels 1 --classes 2 --amp 
--patch-size 32 --networks ming_net trans_unet unet --save-results --augment
```
- 预测
```commandline
python predict.py --model best.pth --epochs 20 --batch-size 8 --learning-rate 1e-5 --scale 1.0 
--channels 1 --classes 2 --amp --patch-size 32 --net-name ming_net --viz --no-save
```
---