import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math


class CosineMoEGate(nn.Module):
    """余弦相似度门控机制，基于现有moe_module.py的实现"""
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, temperature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # 余弦投影层
        self.cosine_projector = nn.Linear(d_model, d_model, bias=False)
        
        # 专家相似矩阵
        self.sim_matrix = nn.Parameter(torch.randn(num_experts, d_model))
        
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.cosine_projector.weight, a=math.sqrt(5))
        nn.init.normal_(self.sim_matrix, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
        
        Returns:
            gate_scores: 门控分数 [batch_size, seq_len, num_experts]
            gate_probs: 门控概率 [batch_size, seq_len, top_k]
            gate_info: 门控信息字典
        """
        batch_size, seq_len, d_model = x.shape
        
        # 余弦投影
        projected = self.cosine_projector(x)  # [batch_size, seq_len, d_model]
        
        # 计算余弦相似度
        projected_norm = F.normalize(projected, p=2, dim=-1)  # [batch_size, seq_len, d_model]
        sim_matrix_norm = F.normalize(self.sim_matrix, p=2, dim=-1)  # [num_experts, d_model]
        
        # 计算相似度分数
        similarity = torch.matmul(projected_norm, sim_matrix_norm.T)  # [batch_size, seq_len, num_experts]
        
        # 应用温度参数
        gate_scores = similarity / self.temperature
        
        # Top-k选择
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # 计算概率
        gate_probs = F.softmax(top_k_scores, dim=-1)  # [batch_size, seq_len, top_k]
        
        # 计算门控信息
        gate_info = self._compute_gate_info(gate_scores, gate_probs, top_k_indices)
        
        return gate_scores, gate_probs, gate_info
    
    def _compute_gate_info(self, gate_scores: torch.Tensor, gate_probs: torch.Tensor, 
                          top_k_indices: torch.Tensor) -> Dict[str, Any]:
        """计算门控信息"""
        batch_size, seq_len, num_experts = gate_scores.shape
        
        # 专家使用情况统计
        expert_usage = torch.zeros(num_experts, device=gate_scores.device)
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, :, i].flatten()
            expert_usage.scatter_add_(0, expert_indices, torch.ones_like(expert_indices, dtype=torch.float))
        
        # 门控概率均值
        gate_probs_mean = gate_probs.mean().item()
        
        gate_info = {
            'temperature': self.temperature.item(),
            'expert_usage': expert_usage.detach().cpu(),
            'gate_probs_mean': gate_probs_mean
        }
        
        return gate_info


class MoEMLP(nn.Module):
    """MoE MLP专家网络"""
    
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2, 
                 temperature: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 门控机制
        self.gate = CosineMoEGate(d_model, num_experts, top_k, temperature)
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        
        # 初始化gate_info
        self.gate_info = {
            'aux_loss': 0.0,
            'expert_usage': None,
            'expert_prob': None,
            'detach_feature': None
        }
    
    def forward(self, x: torch.Tensor, detach_feature: bool = False) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
            detach_feature: 是否分离特征用于备份
        
        Returns:
            output: 输出特征 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 备份原始特征（如果需要）
        if detach_feature:
            self.gate_info['detach_feature'] = x.detach().clone()
        
        # 门控计算
        gate_scores, gate_probs, gate_info = self.gate(x)
        
        # 获取top-k专家索引
        _, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # 初始化输出
        output = torch.zeros_like(x)
        
        # 对每个top-k专家进行计算
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, :, k]  # [batch_size, seq_len]
            expert_probs = gate_probs[:, :, k:k+1]   # [batch_size, seq_len, 1]
            
            # 为每个专家处理输入
            for expert_idx in range(self.num_experts):
                # 找到使用当前专家的位置
                mask = (expert_indices == expert_idx).unsqueeze(-1)  # [batch_size, seq_len, 1]
                
                if mask.any():
                    # 专家计算
                    expert_output = self.experts[expert_idx](x)
                    
                    # 加权累加到输出
                    output += expert_output * expert_probs * mask.float()
        
        # 计算辅助损失（负载均衡损失）
        aux_loss = self._compute_aux_loss(gate_scores)
        
        # 更新gate_info
        self.gate_info.update({
            'aux_loss': aux_loss,
            'expert_usage': gate_info['expert_usage'],
            'expert_prob': gate_info['gate_probs_mean']
        })
        
        return output
    
    def _compute_aux_loss(self, gate_scores: torch.Tensor) -> torch.Tensor:
        """计算辅助损失（负载均衡损失）"""
        # 计算每个专家的平均门控分数
        expert_importance = gate_scores.mean(dim=[0, 1])  # [num_experts]
        
        # 计算负载均衡损失（L2损失）
        target_importance = torch.ones_like(expert_importance) / self.num_experts
        aux_loss = F.mse_loss(expert_importance, target_importance)
        
        return aux_loss
    
    def get_gate_info(self) -> Dict[str, Any]:
        """获取门控信息"""
        return self.gate_info.copy()
    
    def reset_gate_info(self):
        """重置门控信息"""
        self.gate_info = {
            'aux_loss': 0.0,
            'expert_usage': None,
            'expert_prob': None,
            'detach_feature': None
        }


class MoEMLPLayer(nn.Module):
    """MoE MLP层，可以替换标准MLP层"""
    
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2,
                 temperature: float = 1.0, dropout: float = 0.1, use_moe: bool = True):
        super().__init__()
        self.use_moe = use_moe
        
        if use_moe:
            self.mlp = MoEMLP(d_model, d_ff, num_experts, top_k, temperature, dropout)
        else:
            # 标准MLP作为fallback
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
    
    def forward(self, x: torch.Tensor, detach_feature: bool = False) -> torch.Tensor:
        if self.use_moe:
            return self.mlp(x, detach_feature)
        else:
            return self.mlp(x)
    
    def get_gate_info(self) -> Optional[Dict[str, Any]]:
        """获取门控信息（仅MoE模式）"""
        if self.use_moe and hasattr(self.mlp, 'get_gate_info'):
            return self.mlp.get_gate_info()
        return None
    
    def reset_gate_info(self):
        """重置门控信息（仅MoE模式）"""
        if self.use_moe and hasattr(self.mlp, 'reset_gate_info'):
            self.mlp.reset_gate_info()