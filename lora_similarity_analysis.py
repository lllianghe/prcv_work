#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA层权重余弦相似度分析脚本
用于计算四个模态(sk, cp, nir, vis)之间LoRA层权重的余弦相似度
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def load_checkpoint(checkpoint_path):
    """
    加载模型检查点
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    print(f"正在加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 检查检查点结构
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    return state_dict

def extract_lora_weights(state_dict):
    """
    从状态字典中提取LoRA权重
    返回按模态和层组织的权重字典
    """
    lora_weights = {
        'sk': {'lora_down': {}, 'lora_up': {}},
        'cp': {'lora_down': {}, 'lora_up': {}},
        'nir': {'lora_down': {}, 'lora_up': {}},
        'vis': {'lora_down': {}, 'lora_up': {}}
    }
    
    modalities = ['sk', 'cp', 'nir', 'vis']
    
    for key, value in state_dict.items():
        # 查找LoRA相关的权重，基于实际的权重命名规则
        if 'lora_down_' in key or 'lora_up_' in key:
            print(f"找到LoRA权重: {key}")
            
            # 解析键名以确定模态和类型
            for modality in modalities:
                if f'lora_down_{modality}' in key:
                    # 提取层名
                    layer_name = extract_layer_name(key)
                    lora_weights[modality]['lora_down'][layer_name] = value.clone()
                    break
                elif f'lora_up_{modality}' in key:
                    layer_name = extract_layer_name(key)
                    lora_weights[modality]['lora_up'][layer_name] = value.clone()
                    break
    
    return lora_weights

def extract_layer_name(key):
    """
    从权重键名中提取层名
    """
    # 移除模态前缀和后缀，提取层信息
    parts = key.split('.')
    layer_parts = []
    
    for part in parts:
        if 'layer' in part or 'attn' in part or 'proj' in part or 'mlp' in part:
            layer_parts.append(part)
    
    return '.'.join(layer_parts) if layer_parts else key

def flatten_weights(weights_dict):
    """
    将权重字典展平为一维向量
    """
    flattened = []
    for layer_weights in weights_dict.values():
        if isinstance(layer_weights, torch.Tensor):
            flattened.append(layer_weights.flatten())
    
    if flattened:
        return torch.cat(flattened)
    else:
        return torch.tensor([])

def calculate_cosine_similarity(lora_weights):
    """
    计算四个模态之间的余弦相似度
    """
    modalities = ['sk', 'cp', 'nir', 'vis']
    results = {
        'lora_down': {},
        'lora_up': {},
        'combined': {}
    }
    
    for weight_type in ['lora_down', 'lora_up']:
        print(f"\n计算 {weight_type} 权重的余弦相似度...")
        
        # 为每个模态准备权重向量
        modality_vectors = {}
        
        for modality in modalities:
            weights = lora_weights[modality][weight_type]
            if weights:
                flattened = flatten_weights(weights)
                if len(flattened) > 0:
                    modality_vectors[modality] = flattened.numpy()
                    print(f"{modality} {weight_type} 权重维度: {flattened.shape}")
                else:
                    print(f"警告: {modality} {weight_type} 权重为空")
            else:
                print(f"警告: 未找到 {modality} {weight_type} 权重")
        
        # 计算余弦相似度矩阵
        if len(modality_vectors) >= 2:
            similarity_matrix = np.zeros((len(modalities), len(modalities)))
            
            for i, mod1 in enumerate(modalities):
                for j, mod2 in enumerate(modalities):
                    if mod1 in modality_vectors and mod2 in modality_vectors:
                        # 确保向量维度一致
                        vec1 = modality_vectors[mod1]
                        vec2 = modality_vectors[mod2]
                        
                        min_len = min(len(vec1), len(vec2))
                        if min_len > 0:
                            vec1_truncated = vec1[:min_len].reshape(1, -1)
                            vec2_truncated = vec2[:min_len].reshape(1, -1)
                            
                            similarity = cosine_similarity(vec1_truncated, vec2_truncated)[0, 0]
                            similarity_matrix[i, j] = similarity
                        else:
                            similarity_matrix[i, j] = 0.0
                    else:
                        similarity_matrix[i, j] = 0.0
            
            results[weight_type] = {
                'matrix': similarity_matrix,
                'modalities': modalities
            }
        else:
            print(f"警告: {weight_type} 权重向量不足，无法计算相似度")
    
    # 计算组合权重的相似度（lora_down + lora_up）
    print("\n计算组合权重的余弦相似度...")
    combined_vectors = {}
    
    for modality in modalities:
        down_weights = lora_weights[modality]['lora_down']
        up_weights = lora_weights[modality]['lora_up']
        
        down_flat = flatten_weights(down_weights)
        up_flat = flatten_weights(up_weights)
        
        if len(down_flat) > 0 and len(up_flat) > 0:
            combined = torch.cat([down_flat, up_flat])
            combined_vectors[modality] = combined.numpy()
            print(f"{modality} 组合权重维度: {combined.shape}")
        elif len(down_flat) > 0:
            combined_vectors[modality] = down_flat.numpy()
        elif len(up_flat) > 0:
            combined_vectors[modality] = up_flat.numpy()
    
    if len(combined_vectors) >= 2:
        similarity_matrix = np.zeros((len(modalities), len(modalities)))
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if mod1 in combined_vectors and mod2 in combined_vectors:
                    vec1 = combined_vectors[mod1]
                    vec2 = combined_vectors[mod2]
                    
                    min_len = min(len(vec1), len(vec2))
                    if min_len > 0:
                        vec1_truncated = vec1[:min_len].reshape(1, -1)
                        vec2_truncated = vec2[:min_len].reshape(1, -1)
                        
                        similarity = cosine_similarity(vec1_truncated, vec2_truncated)[0, 0]
                        similarity_matrix[i, j] = similarity
                    else:
                        similarity_matrix[i, j] = 0.0
                else:
                    similarity_matrix[i, j] = 0.0
        
        results['combined'] = {
            'matrix': similarity_matrix,
            'modalities': modalities
        }
    
    return results

def print_similarity_results(results):
    """
    打印相似度结果
    """
    modalities = ['sk', 'cp', 'nir', 'vis']
    
    for weight_type, result in results.items():
        if 'matrix' in result:
            print(f"\n=== {weight_type.upper()} 权重余弦相似度矩阵 ===")
            print("\t" + "\t".join(modalities))
            
            matrix = result['matrix']
            for i, mod1 in enumerate(modalities):
                row_str = f"{mod1}\t"
                for j, mod2 in enumerate(modalities):
                    row_str += f"{matrix[i, j]:.4f}\t"
                print(row_str)
            
            # 打印非对角线元素的统计信息
            off_diagonal = []
            for i in range(len(modalities)):
                for j in range(len(modalities)):
                    if i != j:
                        off_diagonal.append(matrix[i, j])
            
            if off_diagonal:
                print(f"\n非对角线元素统计:")
                print(f"平均值: {np.mean(off_diagonal):.4f}")
                print(f"标准差: {np.std(off_diagonal):.4f}")
                print(f"最小值: {np.min(off_diagonal):.4f}")
                print(f"最大值: {np.max(off_diagonal):.4f}")

def plot_similarity_heatmap(results, save_dir=None):
    """
    绘制相似度热力图
    """
    modalities = ['sk', 'cp', 'nir', 'vis']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('LoRA权重余弦相似度热力图', fontsize=16)
    
    weight_types = ['lora_down', 'lora_up', 'combined']
    titles = ['LoRA Down权重', 'LoRA Up权重', '组合权重']
    
    for idx, (weight_type, title) in enumerate(zip(weight_types, titles)):
        if weight_type in results and 'matrix' in results[weight_type]:
            matrix = results[weight_type]['matrix']
            
            sns.heatmap(matrix, 
                       annot=True, 
                       fmt='.3f', 
                       xticklabels=modalities, 
                       yticklabels=modalities,
                       cmap='coolwarm',
                       center=0,
                       ax=axes[idx])
            axes[idx].set_title(title)
        else:
            axes[idx].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(title)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'lora_similarity_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存到: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='分析LoRA层权重的余弦相似度')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--save_dir', type=str, default='./lora_analysis_results', help='结果保存目录')
    
    args = parser.parse_args()
    
    try:
        # 加载检查点
        state_dict = load_checkpoint(args.checkpoint)
        
        # 提取LoRA权重
        print("\n提取LoRA权重...")
        lora_weights = extract_lora_weights(state_dict)
        
        # 检查提取的权重
        print("\n=== LoRA权重提取结果 ===")
        for modality in ['sk', 'cp', 'nir', 'vis']:
            down_count = len(lora_weights[modality]['lora_down'])
            up_count = len(lora_weights[modality]['lora_up'])
            print(f"{modality}: lora_down层数={down_count}, lora_up层数={up_count}")
        
        # 计算余弦相似度
        print("\n计算余弦相似度...")
        similarity_results = calculate_cosine_similarity(lora_weights)
        
        # 打印结果
        print_similarity_results(similarity_results)
        
        # 绘制热力图
        plot_similarity_heatmap(similarity_results, args.save_dir)
        
        print(f"\n分析完成！结果已保存到: {args.save_dir}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()