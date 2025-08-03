DATASET_NAME="ORBench"

# 启用P2P并忽略警告
export NCCL_IGNORE_DISABLED_P2P=1
# 优化NCCL通信设置
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export CUDA_VISIBLE_DEVICES=7  # assign specific GPU
NUM_GPUS=1
torchrun --nproc_per_node=$NUM_GPUS --master_port=54198 \
train.py \
--batch_size 12 \
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