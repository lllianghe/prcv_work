#!/bin/bash


DATASET_NAME="ORBench"


CUDA_VISIBLE_DEVICES=2 \
python train.py \
--batch_size 12 \
--loss_name 'multi_modal_contrastive+itc' \
--sampler random \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/fgclip_large/model.safetensors' \
--test_size 0.125 \
--eval_period 20 \
--drop_last 1 \
--img_aug \
--MLM \
--dataset_name $DATASET_NAME \
--name fgclip_projection_lr \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--num_epoch 700 \
--lr 2.4e-5 \
--warmup_epochs 580 \
--lrscheduler exp \
--power 0.5 \
--step_size 2000 \
--freeze_embedding_layers \
--img_size 384,128 \


echo "训练完成"