#!/bin/bash

# 梯度累积训练示例脚本
# 使用梯度累积功能，每2个step进行一次optimizer.step()

DATASET_NAME="ORBench"

CUDA_VISIBLE_DEVICES=3 \
python train.py \
--batch_size 12 \
--gradient_accumulation_steps 2 \
--drop_last 1 \
--add_multimodal_layers \
--loss_name 'multi_modal_contrastive+itc' \
--sampler random \
--num_instance 5 \
--lr 2.4e-5 \
--warmup_epochs 580 \
--lrscheduler exp \
--step_size 2000 \
--power 0.5 \
--annealing_epochs 4640 \
--min_lr 1e-7 \
--num_epoch 700 \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/fgclip_large/model.safetensors' \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--test_size 0.125 \
--eval_period 20 \
--name large_fgclip_gradient_accumulation \
--img_aug \
--MLM \
--dataset_name $DATASET_NAME

# 说明：
# --gradient_accumulation_steps 2: 每2个step进行一次optimizer.step()
# 这样可以模拟更大的batch size，同时保持学习率曲线不变
# 实际的有效batch size = batch_size * gradient_accumulation_steps = 12 * 2 = 24