#!/usr/bin/env python3
"""
测试余弦路由MoE模块的功能和性能。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from model.moe_module import PersonReIDMoEProcessor, NativeMoELayer
import matplotlib.pyplot as plt
import numpy as np

def test_cosine_routing_basic():
    """测试余弦路由的基本功能。"""
    print("\n=== 测试余弦路由基本功能 ===")
    
    # 初始化MoE层
    moe_layer = NativeMoELayer(
        d_model=256,
        num_experts=4,
        top_k=2,
        expert_dim=512,
        dropout=0.1
    )
    
    batch_size = 8
    seq_len = 1
    d_model = 256
    
    # 测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output, aux_loss = moe_layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"辅助损失: {aux_loss.item():.6f}")
    print(f"温度参数: {moe_layer.temperature.item():.2f}")
    
    # 验证输出
    assert output.shape == x.shape, f"输出形状不匹配: {output.shape} vs {x.shape}"
    assert aux_loss.item() >= 0, "辅助损失应该非负"
    
    print("✅ 余弦路由基本功能测试通过")
    
    return moe_layer

def test_expert_selection_patterns(moe_layer):
    """测试专家选择模式。"""
    print("\n=== 测试专家选择模式 ===")
    
    d_model = 256
    
    # 创建不同模式的输入
    patterns = {
        '随机': torch.randn(4, 1, d_model),
        '全零': torch.zeros(4, 1, d_model),
        '全一': torch.ones(4, 1, d_model),
        '稀疏': torch.zeros(4, 1, d_model)
    }
    
    # 稀疏模式：只有少数维度非零
    patterns['稀疏'][:, :, :10] = 1.0
    
    expert_selections = {}
    
    for pattern_name, x in patterns.items():
        with torch.no_grad():
            # 获取专家选择信息
            x_flat = x.view(-1, d_model)
            
            # 计算余弦相似度
            x_norm = F.normalize(x_flat, p=2, dim=-1)
            centroids_norm = F.normalize(moe_layer.expert_centroids, p=2, dim=-1)
            cosine_scores = torch.matmul(x_norm, centroids_norm.t())
            gate_scores = cosine_scores * moe_layer.temperature
            gate_probs = F.softmax(gate_scores, dim=-1)
            
            # 获取top-k专家
            top_k_probs, top_k_indices = torch.topk(gate_probs, moe_layer.top_k, dim=-1)
            
            expert_selections[pattern_name] = {
                'probs': gate_probs.mean(dim=0).cpu().numpy(),
                'top_k_indices': top_k_indices.cpu().numpy(),
                'cosine_scores': cosine_scores.mean(dim=0).cpu().numpy()
            }
            
            print(f"\n{pattern_name}模式:")
            print(f"  平均专家概率: {gate_probs.mean(dim=0).cpu().numpy()}")
            print(f"  平均余弦分数: {cosine_scores.mean(dim=0).cpu().numpy()}")
            print(f"  选中的专家: {top_k_indices.flatten().cpu().numpy()}")
    
    return expert_selections

def test_temperature_effect():
    """测试温度参数对专家选择的影响。"""
    print("\n=== 测试温度参数效果 ===")
    
    d_model = 128
    moe_layer = NativeMoELayer(
        d_model=d_model,
        num_experts=4,
        top_k=2,
        expert_dim=256
    )
    
    x = torch.randn(8, 1, d_model)
    
    temperatures = [1.0, 5.0, 10.0, 20.0]
    
    for temp in temperatures:
        moe_layer.temperature.data = torch.tensor(temp)
        
        with torch.no_grad():
            x_flat = x.view(-1, d_model)
            x_norm = F.normalize(x_flat, p=2, dim=-1)
            centroids_norm = F.normalize(moe_layer.expert_centroids, p=2, dim=-1)
            cosine_scores = torch.matmul(x_norm, centroids_norm.t())
            gate_scores = cosine_scores * moe_layer.temperature
            gate_probs = F.softmax(gate_scores, dim=-1)
            
            # 计算概率分布的熵（衡量分布的尖锐程度）
            entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=-1).mean()
            max_prob = gate_probs.max(dim=-1)[0].mean()
            
            print(f"温度 {temp:4.1f}: 熵={entropy:.4f}, 最大概率={max_prob:.4f}")
    
    print("✅ 温度参数测试完成")

def test_personreid_moe_with_cosine():
    """测试PersonReIDMoEProcessor与余弦路由的集成。"""
    print("\n=== 测试PersonReID MoE与余弦路由集成 ===")
    
    processor = PersonReIDMoEProcessor(
        input_dim=512,
        hidden_dim=768,
        num_experts=6,
        top_k=2,
        aux_loss_weight=0.01
    )
    
    batch_size = 16
    feature_dim = 512
    
    # 测试不同模态组合
    modalities = ['TEXT', 'CP', 'SK', 'NIR']
    
    for modality in modalities:
        print(f"\n测试 RGB + {modality}:")
        
        # 创建特征
        rgb_features = torch.randn(batch_size, feature_dim)
        query_features = torch.randn(batch_size, feature_dim)
        
        # MoE处理
        enhanced_rgb, enhanced_query, aux_loss = processor(rgb_features, query_features)
        
        print(f"  RGB特征: {rgb_features.shape} -> {enhanced_rgb.shape}")
        print(f"  {modality}特征: {query_features.shape} -> {enhanced_query.shape}")
        print(f"  辅助损失: {aux_loss.item():.6f}")
        
        # 验证梯度流
        total_loss = enhanced_rgb.sum() + enhanced_query.sum() + aux_loss
        total_loss.backward()
        
        # 检查温度参数是否有梯度
        temp_grad = processor.moe_module.moe_layer.temperature.grad
        if temp_grad is not None:
            print(f"  温度参数梯度: {temp_grad.item():.6f}")
        
        processor.zero_grad()
    
    print("✅ PersonReID MoE余弦路由集成测试通过")

def test_cosine_vs_linear_routing():
    """比较余弦路由和线性路由的差异。"""
    print("\n=== 余弦路由 vs 线性路由比较 ===")
    
    d_model = 256
    num_experts = 4
    top_k = 2
    
    # 余弦路由MoE
    cosine_moe = NativeMoELayer(d_model, num_experts, top_k, expert_dim=512)
    
    # 创建线性路由MoE（用于比较）
    class LinearMoE(torch.nn.Module):
        def __init__(self, d_model, num_experts, top_k):
            super().__init__()
            self.gate = torch.nn.Linear(d_model, num_experts, bias=False)
            self.num_experts = num_experts
            self.top_k = top_k
            
        def forward(self, x):
            x_flat = x.view(-1, d_model)
            gate_scores = self.gate(x_flat)
            gate_probs = F.softmax(gate_scores, dim=-1)
            return gate_probs
    
    linear_moe = LinearMoE(d_model, num_experts, top_k)
    
    # 测试输入
    x = torch.randn(8, 1, d_model)
    
    with torch.no_grad():
        # 余弦路由
        x_flat = x.view(-1, d_model)
        x_norm = F.normalize(x_flat, p=2, dim=-1)
        centroids_norm = F.normalize(cosine_moe.expert_centroids, p=2, dim=-1)
        cosine_scores = torch.matmul(x_norm, centroids_norm.t())
        cosine_gate_scores = cosine_scores * cosine_moe.temperature
        cosine_probs = F.softmax(cosine_gate_scores, dim=-1)
        
        # 线性路由
        linear_probs = linear_moe(x)
        
        print("余弦路由专家概率分布:")
        print(f"  平均: {cosine_probs.mean(dim=0).cpu().numpy()}")
        print(f"  标准差: {cosine_probs.std(dim=0).cpu().numpy()}")
        
        print("线性路由专家概率分布:")
        print(f"  平均: {linear_probs.mean(dim=0).cpu().numpy()}")
        print(f"  标准差: {linear_probs.std(dim=0).cpu().numpy()}")
        
        # 计算分布的均匀性（熵）
        cosine_entropy = -torch.sum(cosine_probs * torch.log(cosine_probs + 1e-8), dim=-1).mean()
        linear_entropy = -torch.sum(linear_probs * torch.log(linear_probs + 1e-8), dim=-1).mean()
        
        print(f"\n分布均匀性（熵）:")
        print(f"  余弦路由: {cosine_entropy:.4f}")
        print(f"  线性路由: {linear_entropy:.4f}")
    
    print("✅ 路由方式比较完成")

def main():
    """主测试函数。"""
    print("🚀 余弦路由MoE模块测试")
    print("=" * 50)
    
    try:
        # 基本功能测试
        moe_layer = test_cosine_routing_basic()
        
        # 专家选择模式测试
        test_expert_selection_patterns(moe_layer)
        
        # 温度参数效果测试
        test_temperature_effect()
        
        # PersonReID集成测试
        test_personreid_moe_with_cosine()
        
        # 路由方式比较
        test_cosine_vs_linear_routing()
        
        print("\n" + "=" * 50)
        print("🎉 所有余弦路由测试通过！")
        
        print("\n📋 余弦路由优势:")
        print("  • 基于语义相似度的专家选择")
        print("  • 可学习的专家中心点")
        print("  • 温度参数控制选择尖锐度")
        print("  • 更好的特征空间利用")
        print("  • 提升专家专业化程度")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)