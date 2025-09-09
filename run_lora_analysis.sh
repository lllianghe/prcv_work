#!/bin/bash

# LoRA权重余弦相似度分析脚本使用示例
# 用于分析四个模态(sk, cp, nir, vis)之间LoRA层权重的余弦相似度

# 设置环境变量
export PYTHONPATH="/SSD_Data01/myf/research/PRCV/fgclip_model/prcv_work_layer_norm:$PYTHONPATH"

# 检查点文件路径（请根据实际情况修改）
CHECKPOINT_PATH="/path/to/your/checkpoint.pth"

# 结果保存目录
SAVE_DIR="./lora_analysis_results"

# 检查检查点文件是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT_PATH"
    echo "请修改CHECKPOINT_PATH变量为正确的检查点文件路径"
    exit 1
fi

# 创建结果保存目录
mkdir -p "$SAVE_DIR"

echo "开始LoRA权重相似度分析..."
echo "检查点路径: $CHECKPOINT_PATH"
echo "结果保存目录: $SAVE_DIR"
echo ""

# 运行分析脚本
python lora_similarity_analysis.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --save_dir "$SAVE_DIR"

echo ""
echo "分析完成！"
echo "结果已保存到: $SAVE_DIR"
echo "热力图文件: $SAVE_DIR/lora_similarity_heatmap.png"