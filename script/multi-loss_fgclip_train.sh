#!/bin/bash

"""
我能否动态设置损失权重? 每次计算损失之后根据不同损失的比值来放大较大的损失, 从而给他更多的提升空间?
(不收敛)
降低初始学习率 ( --lr )
让学习率衰减得更早、更快（调整 --milestones 和 --gamma ）。
(过拟合)
正则化




--loss_name 'multi_modal_contrastive+sdm' \
--sk_loss_weight 1.0 \
--nir_loss_weight 4.0 \
--cp_loss_weight 0.8 \
--text_loss_weight 2.0 \
--lr \
--weight declay \


"""


DATASET_NAME="ORBench"

# --resume --resume_ckpt_file '' \
# --weight_decay 1e-3 \
#  --lr 1e-4 \


CUDA_VISIBLE_DEVICES=7 \
python train.py \
--loss_name 'multi_modal_contrastive+itc' \
--test_size 0.375 \
--eval_period 1 \
--val_start_epoch 2 \
--batch_size 28 \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/huggingface_model/model.safetensors' \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--name irra \
--img_aug \
--MLM \
--dataset_name $DATASET_NAME \
--num_epoch 60 \
--val_dataset 'val' \