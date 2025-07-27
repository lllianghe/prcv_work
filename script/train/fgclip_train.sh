#!/bin/bash
DATASET_NAME="ORBench"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name irra \
--img_aug \
--batch_size 16 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+id' \
--num_epoch 60 \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/fgclip/model.safetensors' \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/'