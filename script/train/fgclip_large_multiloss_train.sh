DATASET_NAME="ORBench"

CUDA_VISIBLE_DEVICES=7 \
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
--name large_fgclip \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--lr 2.4e-5 \
--warmup_epochs 580 \
--lrscheduler cosine \
--annealing_epochs 4640 \
--min_lr 1e-7 \
--num_epoch 700 \
--resume \
--resume_ckpt_file '/SSD_Data01/myf/research/PRCV/fgclip_model/prcv_work/logs/ORBench/20250806_062658_large_fgclip/best.pth' \

# --power 0.5 \
# --step_size 2000 \
# --add_multimodal_layers \

# --img_size 336,336 \
# --autocast_dtype torch.float16 \


# --num_instance 5 \
# --gradient_accumulation_steps 1 \
#

# 按照epoch来调整log_period和scheduler_period
# --log_period 20 \
# --scheduler_period 20 \
