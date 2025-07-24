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


CUDA_VISIBLE_DEVICES=7 \
python train.py \
--loss_name 'multi_modal_contrastive+itc' \
--lr 5e-6 \
--lrscheduler 'cosine_warm' \
--optimizer Adam \
--schedule_steps 50 \
--warmup_steps 400 \
--annealing_steps 2000 \
--test_size 0.125 \
--eval_period 1 \
--val_start_epoch 2 \
--batch_size 30 \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/huggingface_model/model.safetensors' \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--name irra \
--img_aug \
--MLM \
--dataset_name $DATASET_NAME \
--val_dataset 'val' \
--num_epoch 20 \