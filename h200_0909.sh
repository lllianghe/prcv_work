set -e  # 任何命令失败时退出脚本
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-hzc
echo "当前 Conda 环境: $CONDA_DEFAULT_ENV"
echo "当前 Python 路径: $(which python)"

# 同步代码
cd /root/worker_gpfs/hzc-comp/prcv_data/prcv_work
git checkout a800
git pull origin a800


# 训练命令
echo "$(date): 开始训练"
CUDA_VISIBLE_DEVICES=6 python train.py \
--batch_size 32 \
--loss_name 'multi_modal_contrastive+itc' \
--sampler random \
--pretrain_choice '/root/worker_gpfs/hzc-comp/prcv_data/model_cache/fgclip_large/model.safetensors' \
--test_size 0.3333 \
--eval_period 20 \
--drop_last 1 \
--img_aug \
--MLM \
--dataset_name "ORBench" \
--name fgclip \
--root_dir '/root/worker_gpfs/hzc-comp/prcv_data' \
--warmup_epochs 580 \
--lrscheduler exp \
--power 0.5 \
--step_size 2000 \
--add_multimodal_layers \
--img_size 336,336 \
--num_epoch 800 \
--lr 2.4e-5 \
--ln_lr 1e-3 \
--weight_decay 4e-5 \
--lora_backbone_lr 1e-6 \
--lora_lr 1e-1 

# 生成kaggle csv
echo "$(date): 开始生成CSV"
CUDA_VISIBLE_DEVICES=6 \
python get_kaggle_csv.py || { echo "生成CSV失败"; exit 1; }

echo "$(date): a800脚本运行完成"