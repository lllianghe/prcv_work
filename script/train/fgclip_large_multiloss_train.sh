DATASET_NAME="ORBench"

CUDA_VISIBLE_DEVICES=5 \
python train.py \
--batch_size 12 \
--drop_last 1 \
--sampler random \
--num_instance 5 \
--loss_name 'multi_modal_contrastive+itc' \
--warmup_epochs 1000 \
--lr 2e-5 \
--annealing_epochs 8000 \
--min_lr 1e-7 \
--num_epoch 700 \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/fgclip_large/model.safetensors' \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--test_size 0.125 \
--eval_period 20 \
--name large_fgclip \
--img_aug \
--MLM \
--dataset_name $DATASET_NAME \



# 按照epoch来调整log_period和scheduler_period
# --log_period 20 \
# --scheduler_period 20 \
