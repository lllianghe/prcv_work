DATASET_NAME="ORBench"

CUDA_VISIBLE_DEVICES=4 \
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
--name fgclip \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--warmup_epochs 580 \
--lrscheduler exp \
--power 0.5 \
--step_size 2000 \
--add_multimodal_layers \
--img_size 384,128 \
--num_epoch 800 \
--lr 2.4e-5 \
--ln_lr 1e-3 \
--weight_decay 4e-5 \
--lora_lr


# --annealing_epochs 4640 \
# --min_lr 1e-7 \

# --img_size 336,336 \
# --autocast_dtype torch.float16 \


# --num_instance 5 \
# --gradient_accumulation_steps 1 \
#

# 按照epoch来调整log_period和scheduler_period
# --log_period 20 \
# --scheduler_period 20 \
