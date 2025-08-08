#!/bin/bash

set -e  # 任何命令失败时退出脚本

#需要关闭代理连接github
# unset $http_proxy
# unset $https_proxy
# 首次克隆仓库
cd /fs-computility/ai-shen/macaoyuan.p/hzc/PRCV
if [ ! -d "prcv_work" ]; then
    git clone -b a800 https://github.com/lllianghe/prcv_work.git
else
    echo "prcv_work目录已存在，跳过克隆"
fi

# 激活conda环境
source /fs-computility/ai-shen/macaoyuan.p/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-hzc
echo "当前 Conda 环境: $CONDA_DEFAULT_ENV"
echo "当前 Python 路径: $(which python)"

# 同步代码
cd /fs-computility/ai-shen/macaoyuan.p/hzc/PRCV/prcv_work
git checkout a800
git pull origin a800

#需要添加代理连接wandb
export http_proxy=http://100.68.170.107:3128
export https_proxy=http://100.68.170.107:3128


# 训练命令
echo "$(date): 开始训练"
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--batch_size 32 \
--drop_last 1 \
--loss_name 'multi_modal_contrastive+itc' \
--sampler random \
--lr 2.4e-5 \
--warmup_epochs 580 \
--num_epoch 700 \
--pretrain_choice '/fs-computility/ai-shen/macaoyuan.p/hzc/PRCV/model_cache/fgclip_large/model.safetensors' \
--root_dir '/fs-computility/ai-shen/macaoyuan.p/hzc/PRCV/' \
--test_size 0 \
--eval_period 20 \
--name large_fgclip \
--img_aug \
--MLM \
--img_size 336,336 \
--add_multimodal_layers \
--lrscheduler exp \
--step_size 2000 \
--power 0.5 || { echo "训练失败"; exit 1; }
echo "$(date): 模型训练运行完成"

# 生成kaggle csv
echo "$(date): 开始生成CSV"
CUDA_VISIBLE_DEVICES=0 \
python get_kaggle_csv.py || { echo "生成CSV失败"; exit 1; }

echo "$(date): a800脚本运行完成"