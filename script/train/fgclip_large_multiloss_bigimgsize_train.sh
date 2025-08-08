DATASET_NAME="ORBench"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--batch_size 10 \
--drop_last 1 \
--loss_name 'multi_modal_contrastive+itc' \
--sampler random \
--lr 2.4e-5 \
--warmup_epochs 580 \
--num_epoch 700 \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/fgclip_large/model.safetensors' \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--test_size 0.125 \
--eval_period 20 \
--name large_fgclip \
--img_aug \
--MLM \
--dataset_name $DATASET_NAME \
--img_size 336,336 \
--add_multimodal_layers \
--lrscheduler exp \
--step_size 2000 \
--power 0.5 \
# --annealing_epochs 4640 \
# --min_lr 1e-7 \

# --num_instance 5 \
# --gradient_accumulation_steps 1 \
# 

# 按照epoch来调整log_period和scheduler_period
# --log_period 20 \
# --scheduler_period 20 \
