#!/bin/bash

"""
我能否动态设置损失权重? 每次计算损失之后根据不同损失的比值来放大较大的损失, 从而给他更多的提升空间?
(不收敛)
降低初始学习率 ( --lr )
让学习率衰减得更早、更快（调整 --milestones 和 --gamma ）。
(过拟合)
正则化


"""


DATASET_NAME="ORBench"

# --resume --resume_ckpt_file '' \
# --optimizer Adamw \
# --args.momentum \
# --weight_decay 1e-3 \


CUDA_VISIBLE_DEVICES=4 \
python train.py \
--batch_size 60 \
--loss_name 'multi_modal_contrastive+sdm' \
--scheduler_period 30 \
--warmup_epochs 1000 \
--lr 5e-6 \
--annealing_epochs 4000 \
--min_lr 1e-7 \
--num_epoch 30 \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/huggingface_model/model.safetensors' \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--test_size 0.125 \
--eval_period 1 \
--name irra \
--img_aug \
--MLM \
--dataset_name $DATASET_NAME \
--log_period 30 \