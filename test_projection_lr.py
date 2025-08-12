#!/usr/bin/env python3
"""
测试projection层学习率因子设置功能

这个脚本演示了如何使用--add_multimodal_projections参数来为projection层设置4倍学习率。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from utils.options import get_args
from model import build_model
from solver import build_optimizer

def test_projection_lr_factor():
    """
    测试projection层学习率因子功能
    """
    print("=" * 60)
    print("测试projection层学习率因子设置功能")
    print("=" * 60)
    
    # 模拟命令行参数
    import argparse
    
    # 创建一个简单的参数对象用于测试
    class TestArgs:
        def __init__(self, add_multimodal_projections=False, add_multimodal_layers=False):
            # 基本参数
            self.lr = 2.4e-5
            self.lr_factor = 5.0
            self.bias_lr_factor = 2.0
            self.weight_decay = 4e-5
            self.weight_decay_bias = 0.0
            self.optimizer = 'Adam'
            self.alpha = 0.9
            self.beta = 0.999
            self.momentum = 0.9
            
            # 多模态相关参数
            self.add_multimodal_projections = add_multimodal_projections
            self.add_multimodal_layers = add_multimodal_layers
            self.add_multimodal_embeddings = False
            
            # 添加其他必要的参数
            self.dataset_name = 'ORBench'
            self.pretrain_choice = 'ViT-B/16'
            self.img_size = (384, 128)
            self.text_length = 77
            self.vocab_size = 49408
            self.cmt_depth = 4
            self.temperature = 0.02
            self.stride_size = 16
            self.masked_token_rate = 0.8
            self.masked_token_unchanged_rate = 0.1
            self.MLM = True
            self.loss_names = 'multi_modal_contrastive+itc'
            self.mlm_loss_weight = 1.0
            self.id_loss_weight = 1.0
            self.img_aug = True
    
    # 测试不启用多模态projection的情况
    print("\n1. 测试不启用--add_multimodal_projections的情况:")
    print("-" * 50)
    
    args_without = TestArgs(add_multimodal_projections=False)
    
    try:
        # 构建模型（简化版本，只用于测试参数名称）
        model = build_model(args_without)
        
        # 构建优化器
        optimizer = build_optimizer(args_without, model)
        
        print("✓ 成功构建优化器（未启用projection层学习率因子）")
        
        # 检查参数组
        projection_params = []
        for group in optimizer.param_groups:
            for param in group['params']:
                # 通过模型找到对应的参数名
                for name, model_param in model.named_parameters():
                    if param is model_param and ('projection' in name):
                        projection_params.append((name, group['lr']))
                        break
        
        if projection_params:
            print(f"找到 {len(projection_params)} 个projection层参数:")
            for name, lr in projection_params:
                print(f"  - {name}: lr = {lr:.2e} (基础学习率)")
        else:
            print("未找到projection层参数")
            
    except Exception as e:
        print(f"✗ 构建失败: {e}")
    
    # 测试启用多模态projection的情况
    print("\n2. 测试启用--add_multimodal_projections的情况:")
    print("-" * 50)
    
    args_with = TestArgs(add_multimodal_projections=True)
    
    try:
        # 构建模型
        model = build_model(args_with)
        
        # 手动设置多模态projection层（模拟训练脚本中的逻辑）
        if hasattr(model.base_model, 'setup_multi_projections'):
            model.base_model.setup_multi_projections()
            print("✓ 已设置多模态projection层")
        
        # 构建优化器
        optimizer = build_optimizer(args_with, model)
        
        print("✓ 成功构建优化器（已启用projection层学习率因子）")
        
        # 检查参数组
        projection_params = []
        other_params = []
        
        for group in optimizer.param_groups:
            for param in group['params']:
                # 通过模型找到对应的参数名
                for name, model_param in model.named_parameters():
                     if param is model_param:
                         if any(proj_key in name for proj_key in ['text_projection', 'modality_visual_projections']):
                             projection_params.append((name, group['lr']))
                         else:
                             other_params.append((name, group['lr']))
                         break
        
        print(f"\n找到 {len(projection_params)} 个projection层参数:")
        for name, lr in projection_params:
            expected_lr = args_with.lr * 4.0
            status = "✓" if abs(lr - expected_lr) < 1e-10 else "✗"
            print(f"  {status} {name}: lr = {lr:.2e} (期望: {expected_lr:.2e})")
        
        print(f"\n其他参数示例 (前5个):")
        for name, lr in other_params[:5]:
            print(f"  - {name}: lr = {lr:.2e}")
            
    except Exception as e:
        print(f"✗ 构建失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    print("\n使用说明:")
    print("1. 在训练脚本中添加 --add_multimodal_projections 参数")
    print("2. 或者使用 --add_multimodal_layers 参数（包含projection和embedding层）")
    print("3. 所有projection层将自动获得4倍学习率")
    print("4. 只包括FGCLIPModel中定义的: text_projection, modality_visual_projections")
    
    print("\n示例命令:")
    print("python train.py --add_multimodal_projections --lr 2.4e-5 [其他参数...]")
    print("# projection层将使用 9.6e-5 的学习率 (2.4e-5 * 4)")

if __name__ == "__main__":
    test_projection_lr_factor()