#!/bin/bash

# 使用projection层学习率因子的训练脚本示例
# 这个脚本演示了如何使用--add_multimodal_projections参数来为projection层设置4倍学习率

DATASET_NAME="ORBench"

echo "开始训练 - 使用projection层4倍学习率"
echo "基础学习率: 2.4e-5"
echo "projection层学习率: 9.6e-5 (4倍)"
echo "======================================"

CUDA_VISIBLE_DEVICES=2 \
python train.py \
--batch_size 24 \
--loss_name 'multi_modal_contrastive+itc' \
--sampler random \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/huggingface_model/model.safetensors' \
--test_size 0.125 \
--eval_period 20 \
--drop_last 1 \
--img_aug \
--MLM \
--dataset_name $DATASET_NAME \
--name fgclip_projection_lr \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--num_epoch 1000 \
--lr 2.4e-5 \
--warmup_epochs 580 \
--lrscheduler exp \
--power 0.5 \
--step_size 2000 \
--add_multimodal_projections \
--img_size 224,224 \

# 注意: --add_multimodal_projections 参数会自动为所有projection层设置4倍学习率
# 包括: visual_projection, text_projection, text_filip_projection, modality_visual_projections

# 如果你想同时添加embedding和projection层，可以使用:
# --add_multimodal_layers \
# 这等价于同时使用 --add_multimodal_embeddings 和 --add_multimodal_projections

echo "训练完成"