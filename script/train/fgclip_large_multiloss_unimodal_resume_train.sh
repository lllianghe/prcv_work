#!/bin/bash

# 配置参数
DATASET_NAME="ORBench"  # 数据集名称
CHECKPOINT_FILE="/SSD_Data01/myf/research/PRCV/fgclip_model/prcv_work/logs/ORBench/20250802_163940_large_fgclip/best.pth"  # 检查点文件路径
ADD_MULTIMODAL_LAYERS=true    # 是否添加多模态层 (true=单模态检查点+自动添加多模态层, false=多模态检查点)

CUDA_VISIBLE_DEVICES=6 \
python train.py \
--batch_size 12 \
--drop_last 1 \
--loss_name 'multi_modal_contrastive+itc' \
--sampler random \
--num_instance 5 \
--warmup_epochs 300 \
--lr 6e-6 \
--lrscheduler exp \
--step_size 2000 \
--power 0.5 \
--milestones 500 700 900 1100 1300 1500 1700 1900 2100 2300 2500 2700 2900 3100 3300 3500 3700 3900 4100 4300 4500 4700 4900 5100 5300 5500 5700 5900 6100 6300 6500 6700 6900 7100 7300 7500 7700 7900 8100 8300 8500 8700 8900 \
--gamma 0.93303299 \
--annealing_epochs 9280 \
--min_lr 1e-9 \
--num_epoch 700 \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/fgclip_large/model.safetensors' \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--test_size 0.125 \
--eval_period 10 \
--name large_fgclip_multimodal_resume \
--img_aug \
--MLM \
--dataset_name $DATASET_NAME \
--resume \
--resume_ckpt_file $CHECKPOINT_FILE \
$(if [ "$ADD_MULTIMODAL_LAYERS" = "true" ]; then echo "--add_multimodal_layers"; fi)

# 使用说明：
# 1. 修改 CHECKPOINT_FILE 变量为你的检查点路径
# 2. 设置 ADD_MULTIMODAL_LAYERS 为 true 如果是单模态检查点(会自动添加多模态层)，false 如果是多模态检查点

# 示例：从单模态检查点继续训练(自动添加多模态层)
# CHECKPOINT_FILE="/path/to/single_modal_checkpoint.pth"
# ADD_MULTIMODAL_LAYERS=true

# 示例：从多模态检查点恢复训练
# CHECKPOINT_FILE="/path/to/multimodal_checkpoint.pth"
# ADD_MULTIMODAL_LAYERS=false