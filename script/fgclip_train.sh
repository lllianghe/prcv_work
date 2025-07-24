#!/bin/bash

"""
我能否动态设置损失权重? 每次计算损失之后根据不同损失的比值来放大较大的损失, 从而给他更多的提升空间?
(不收敛)
降低初始学习率 ( --lr )
让学习率衰减得更早、更快（调整 --milestones 和 --gamma ）。
(过拟合)
正则化

--sk_loss_weight 1.0 \
--nir_loss_weight 4.0 \
--cp_loss_weight 0.8 \
--text_loss_weight 2.0 \

--sk_loss_weight 0.5 \
--nir_loss_weight 4.0 \
--cp_loss_weight 0.25 \
--text_loss_weight 2.0 \
"""


DATASET_NAME="ORBench"

# --resume --resume_ckpt_file '' \
# --test_size 0.0 \
# --loss_name 'multi_modal_contrastive+itc+sdm' \
CUDA_VISIBLE_DEVICES=1 \
python train.py \
--loss_name 'sdm+id' \
--test_size 0.125 \
--eval_period 1 \
--val_start_epoch 5 \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/huggingface_model/model.safetensors' \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--name irra \
--img_aug \
--batch_size 30 \
--MLM \
--dataset_name $DATASET_NAME \
--num_epoch 60 \
--val_dataset 'val' \