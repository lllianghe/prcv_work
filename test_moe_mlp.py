#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试MoE MLP层替换功能
验证模型前向传播和维度匹配是否正确
"""

import torch
import torch.nn as nn
from model.build import IRRA
from model.moe_mlp import MoEMLPLayer
import argparse

def create_test_args():
    """创建测试用的参数配置"""
    args = argparse.Namespace()
    
    # 基本模型参数
    args.dataset_name = "test"
    args.pretrain_choice = "ViT-B/16"
    args.img_size = (224, 224)
    args.stride_size = 16
    args.temperature = 0.07
    args.loss_names = "id+itc"
    args.vocab_size = 49408
    args.cmt_depth = 4
    
    # MoE相关参数
    args.moe_num_experts = 4
    args.moe_top_k = 2
    args.moe_modal_aux_loss_weight = 0.1
    args.moe_global_aux_loss_weight = 0.1
    
    # MoE MLP层替换参数
    args.global_aux_loss_weight = 1.0
    args.modal_aux_loss_weight = 1.0
    args.moe_mlp_layers = 6  # 启用MoE MLP替换
    args.moe_mlp_lr = 1e-4
    
    return args

def test_model_structure(model):
    """测试模型结构是否正确"""
    print("\n=== 测试模型结构 ===")
    
    # 检查vision encoder的后6层是否已替换为MoE MLP
    vision_encoder = model.base_model.vision_model.encoder
    total_layers = len(vision_encoder.layers)
    print(f"Vision encoder总层数: {total_layers}")
    
    # 检查后6层（索引6-11）
    replaced_layers = []
    for i in range(6, min(12, total_layers)):
        layer = vision_encoder.layers[i]
        if isinstance(layer.mlp, MoEMLPLayer):
            replaced_layers.append(i)
            print(f"第{i}层MLP已替换为MoE MLP")
        else:
            print(f"第{i}层MLP仍为原始CLIP MLP")
    
    print(f"成功替换的层数: {len(replaced_layers)}")
    return len(replaced_layers) > 0

def test_forward_pass(model, batch_size=2):
    """测试前向传播"""
    print("\n=== 测试前向传播 ===")
    
    # 创建测试数据
    device = next(model.parameters()).device
    
    # 图像数据 (batch_size, 3, 224, 224)
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 文本数据 (batch_size, 77) - CLIP的标准文本长度
    text_tokens = torch.randint(0, 49408, (batch_size, 77)).to(device)
    
    try:
        # 测试图像编码
        print("测试图像编码...")
        with torch.no_grad():
            image_features = model.encode_image(images)
        print(f"图像特征维度: {image_features.shape}")
        
        # 测试文本编码
        print("测试文本编码...")
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
        print(f"文本特征维度: {text_features.shape}")
        
        # 检查维度匹配
        if image_features.shape[-1] == text_features.shape[-1]:
            print("✓ 图像和文本特征维度匹配")
            return True
        else:
            print("✗ 图像和文本特征维度不匹配")
            return False
            
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False

def test_moe_functionality(model):
    """测试MoE功能"""
    print("\n=== 测试MoE功能 ===")
    
    # 检查是否有MoE层
    vision_encoder = model.base_model.vision_model.encoder
    moe_layers = []
    
    for i, layer in enumerate(vision_encoder.layers):
        if isinstance(layer.mlp, MoEMLPLayer):
            moe_layers.append((i, layer.mlp))
    
    if not moe_layers:
        print("未找到MoE层")
        return False
    
    print(f"找到{len(moe_layers)}个MoE层")
    
    # 测试MoE层的门控信息
    device = next(model.parameters()).device
    test_input = torch.randn(2, 3, 224, 224).to(device)
    
    try:
        with torch.no_grad():
            _ = model.encode_image(test_input)
        
        # 检查门控信息
        for layer_idx, moe_layer in moe_layers:
            gate_info = moe_layer.get_gate_info()
            if gate_info:
                print(f"第{layer_idx}层MoE门控信息: {len(gate_info)}个批次")
            else:
                print(f"第{layer_idx}层MoE门控信息为空")
        
        return True
        
    except Exception as e:
        print(f"✗ MoE功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试MoE MLP层替换功能...")
    
    # 创建测试参数
    args = create_test_args()
    
    try:
        # 创建模型
        print("创建IRRA模型...")
        model = IRRA(args, num_classes=1000)
        model.eval()
        
        # 如果有GPU，移动到GPU
        if torch.cuda.is_available():
            model = model.cuda()
            print("模型已移动到GPU")
        
        # 运行测试
        structure_ok = test_model_structure(model)
        forward_ok = test_forward_pass(model)
        moe_ok = test_moe_functionality(model)
        
        # 总结测试结果
        print("\n=== 测试结果总结 ===")
        print(f"模型结构测试: {'✓ 通过' if structure_ok else '✗ 失败'}")
        print(f"前向传播测试: {'✓ 通过' if forward_ok else '✗ 失败'}")
        print(f"MoE功能测试: {'✓ 通过' if moe_ok else '✗ 失败'}")
        
        if structure_ok and forward_ok and moe_ok:
            print("\n🎉 所有测试通过！MoE MLP层替换功能正常工作")
        else:
            print("\n❌ 部分测试失败，请检查实现")
            
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()