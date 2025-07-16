#!/bin/bash
DATASET_NAME="ORBench"

CUDA_VISIBLE_DEVICES=1 \
python train.py \
--name irra \
--img_aug \
--batch_size 32 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+id' \
--num_epoch 60 \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/'