DATASET_NAME="ORBench"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--batch_size 14 \
--drop_last 1 \
--loss_name 'multi_modal_contrastive+itc' \
--sampler random \
--num_instance 5 \
--gradient_accumulation_steps 1 \
--warmup_epochs 580 \
--lr 2e-5 \
--lrscheduler cosine \
--step_size 10000 \
--power 0.5 \
--annealing_epochs 4640 \
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

# --add_multimodal_layers \

# 按照epoch来调整log_period和scheduler_period
# --log_period 20 \
# --scheduler_period 20 \
