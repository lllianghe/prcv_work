DATASET_NAME="ORBench"

CUDA_VISIBLE_DEVICES=5 \
python train.py \
--batch_size 12 \
--sampler random \
--num_instance 4 \
--loss_name 'multi_modal_contrastive+itc' \
--scheduler_period 30 \
--warmup_epochs 1000 \
--lr 5e-6 \
--annealing_epochs 8000 \
--min_lr 1e-7 \
--num_epoch 90 \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/fgclip_large/model.safetensors' \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--test_size 0.125 \
--eval_period 1 \
--name large_fgclip \
--img_aug \
--MLM \
--dataset_name $DATASET_NAME \
--log_period 30 \