import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Disable tutel due to GPU architecture compatibility issues
# Use native PyTorch implementation instead
tutel_moe = None
print("Info: Using native PyTorch MoE implementation for better compatibility.")


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class NativeMoELayer(nn.Module):
    """Native PyTorch implementation of Mixture of Experts layer with cosine routing.
    
    This implementation follows tutel's cosine routing mechanism for consistency:
    - Uses cosine projector + similarity matrix (like tutel's CosineTopKGate)
    - Temperature parameter with clamping
    - Proper initialization matching tutel's approach
    """
    def __init__(self, d_model, num_experts, top_k, expert_dim=None, dropout=0.0, proj_dim=256, init_t=0.5):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(num_experts, top_k)  # Ensure top_k doesn't exceed num_experts
        self.expert_dim = expert_dim or d_model * 4
        
        # Cosine routing components (following tutel's CosineTopKGate)
        # Temperature parameter (log-parameterized like tutel)
        self.temperature = nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        
        # Cosine projector to reduce dimensionality
        self.cosine_projector = nn.Linear(d_model, proj_dim)
        
        # Similarity matrix for expert selection
        self.sim_matrix = nn.Parameter(torch.randn(size=(proj_dim, num_experts)), requires_grad=True)
        
        # Temperature clamping (following tutel)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        
        # Initialize similarity matrix (following tutel)
        nn.init.normal_(self.sim_matrix, 0, 0.01)
        
        # Expert networks - using RepAdapter experts
        self.experts = nn.ModuleList([
            RepAdapter(
                d_model=d_model,
                bottleneck=self.expert_dim,
                dropout=dropout,
                init_option="lora",
                adapter_scalar=1.0
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Enhanced features
            aux_loss: Auxiliary loss for load balancing
            gate_info: Dictionary containing gate probabilities and expert usage
        """
        batch_size, seq_len, d_model = x.shape
        
        # Flatten for processing
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)
        
        # Cosine routing following tutel's CosineTopKGate implementation
        # Project input through cosine projector
        projected_x = self.cosine_projector(x_flat)  # (batch_size * seq_len, proj_dim)
        
        # Compute cosine similarity with similarity matrix
        # Normalize projected features and similarity matrix
        projected_x_norm = F.normalize(projected_x, dim=1)  # (batch_size * seq_len, proj_dim)
        sim_matrix_norm = F.normalize(self.sim_matrix, dim=0)  # (proj_dim, num_experts)
        
        # Compute logits via matrix multiplication
        logits = torch.matmul(projected_x_norm, sim_matrix_norm)  # (batch_size * seq_len, num_experts)
        
        # Apply temperature scaling with clamping (following tutel)
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        gate_scores = logits * logit_scale
        gate_probs = F.softmax(gate_scores, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Collect gate information for logging
        gate_info = {
            'gate_probs_mean': gate_probs.mean(dim=0),  # Average probability for each expert (keep gradients)
            'expert_usage': torch.bincount(top_k_indices.flatten(), minlength=self.num_experts).float().detach().cpu(),  # Usage count per expert
            'temperature': logit_scale.detach().cpu().item()          # Current temperature value
        }
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process through selected experts
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]  # (batch_size * seq_len,)
            expert_probs = top_k_probs[:, i:i+1]  # (batch_size * seq_len, 1)
            
            # Group inputs by expert
            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]  # (num_tokens_for_expert, d_model)
                    expert_output = self.experts[expert_id](expert_input)
                    
                    # Add weighted expert output
                    output[mask] += expert_probs[mask] * expert_output
        
        # Compute auxiliary loss for load balancing
        aux_loss = self._compute_aux_loss(gate_probs)
        
        # Reshape back to original shape
        output = output.view(batch_size, seq_len, d_model)
        
        return output, aux_loss, gate_info
    
    def _compute_aux_loss(self, gate_probs):
        """Compute auxiliary loss for load balancing."""
        # Compute the fraction of tokens assigned to each expert
        expert_counts = gate_probs.sum(dim=0)  # (num_experts,)
        total_tokens = gate_probs.size(0)
        
        # Compute load balancing loss
        # Encourage uniform distribution across experts
        uniform_prob = 1.0 / self.num_experts
        expert_fractions = expert_counts / total_tokens
        
        # L2 loss between actual and uniform distribution
        aux_loss = torch.sum((expert_fractions - uniform_prob) ** 2)
        
        return aux_loss


class RepAdapter(nn.Module):
    """RepAdapter: Representation Adapter for multi-modal feature processing.
    
    This module uses 1D convolutions to process input features and applies
    residual connections for stable training. Compatible with MFRNet implementation.
    """
    def __init__(self, d_model, bottleneck=512, dropout=0.1, init_option="lora", adapter_scalar=1.0):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.adapter_scalar = adapter_scalar
        self.dropout = dropout
            
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            # Use Conv1d layers as in the original implementation
            self.conv_A = nn.Conv1d(self.n_embd, self.down_size, 1, groups=1, bias=True)
            self.conv_B = nn.Conv1d(self.down_size, self.n_embd, 1, groups=1, bias=True)
            
            # Initialize weights using Xavier uniform
            nn.init.xavier_uniform_(self.conv_A.weight)
            nn.init.xavier_uniform_(self.conv_B.weight)
            nn.init.zeros_(self.conv_A.bias)
            nn.init.zeros_(self.conv_B.bias)
        else:
            raise NotImplementedError
            
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # 2D input: (num_tokens, d_model) -> add sequence dimension
            x = x.unsqueeze(0)  # (1, num_tokens, d_model)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Transpose for Conv1d: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        residual = x
        
        # Apply convolutions with residual connection (MFRNet style)
        result = self.conv_A(x)
        result = self.dropout_layer(result)
        result = self.conv_B(result)
        result = result * self.adapter_scalar + residual
        
        # Transpose back: (batch, features, seq_len) -> (batch, seq_len, features)
        result = result.transpose(1, 2).contiguous()
        
        # Remove added dimension if input was 2D
        if squeeze_output:
            result = result.squeeze(0)  # (num_tokens, d_model)
            
        return result


class MultiModalMoEModule(nn.Module):
    """Multi-Modal Mixture of Experts Module for Person Re-identification.
    
    This module applies MoE mechanism to multi-modal features (vis, cp, sk, nir)
    to extract common and differentiated features across modalities.
    """
    def __init__(self, 
                 d_model: int = 512,
                 num_experts: int = 6,
                 top_k: int = 2,
                 gate_type: str = "cosine_top",
                 bottleneck: Optional[int] = None,
                 dropout: float = 0.0,
                 use_moe: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_moe = use_moe
        self.bottleneck = bottleneck
        
        # Set expert hidden dimension
        expert_dim = bottleneck if bottleneck is not None else d_model * 4
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        if use_moe:
            print("Using MoE")
            # Use native PyTorch MoE implementation for better compatibility
            self.moe_layer = NativeMoELayer(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k,
                expert_dim=expert_dim,
                dropout=dropout
            )
            print(f"Initialized native MoE with cosine routing: {num_experts} experts, top-{top_k} selection, expert_dim={expert_dim}")
        else:
            # Fallback: use a simple adapter if MoE is not available
            print("Warning: Using fallback adapter instead of MoE")
            self.fallback_adapter = RepAdapter(d_model, bottleneck, dropout)
            
    def forward(self, features, modality_mask=None):
        """
        Args:
            features: Input features of shape (batch_size, seq_len, d_model)
            modality_mask: Optional mask indicating which modalities are present
            
        Returns:
            Enhanced features and auxiliary loss (if using MoE)
        """
        # Apply layer normalization
        norm_features = self.layer_norm(features)
        



        if self.use_moe and hasattr(self, 'moe_layer'):
            # Apply native MoE processing
            enhanced_features, aux_loss, gate_info = self.moe_layer(norm_features)
            
            # Add residual connection
            output = features + enhanced_features
            
            return output, aux_loss, gate_info
        else:
            # Use fallback adapter
            print(f"use fallback_layers")
            enhanced_features = self.fallback_adapter(norm_features)
            
            # Add residual connection
            output = features + enhanced_features
            
            # Return empty gate info for fallback
            empty_gate_info = {
                'gate_probs_mean': torch.zeros(1),
                'expert_usage': torch.zeros(1),
                'temperature': 1.0
            }
            
            return output, torch.tensor(0.0, device=features.device), empty_gate_info
            
    def get_aux_loss_weight(self):
        """Get the auxiliary loss weight for MoE training."""
        return 0.01  # Standard auxiliary loss weight


class PersonReIDMoEProcessor(nn.Module):
    """Person Re-ID specific MoE processor that handles two modalities at a time.
    
    This module processes RGB gallery features + one query modality features
    and applies MoE to extract enhanced representations for efficient training.
    """
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dim: int = 1024,
                 num_experts: int = 16,
                 top_k: int = 6,
                 aux_loss_weight: float = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        
        # MoE module for processing two modalities
        self.moe_module = MultiModalMoEModule(
            d_model=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            bottleneck=hidden_dim,
            dropout=dropout,
            use_moe=True
        )
        
        print(f"PersonReIDMoEProcessor initialized: {num_experts} experts, top-{top_k}, aux_weight={aux_loss_weight}")
        
    def forward(self, rgb_features, query_features):
        """
        Process two modalities (RGB + query) through MoE.
        
        Args:
            rgb_features: RGB gallery features of shape (batch_size, feature_dim)
            query_features: Query modality features of shape (batch_size, feature_dim)
                          
        Returns:
            Tuple of (enhanced_rgb_features, enhanced_query_features, aux_loss, gate_info)
        """
        # Add sequence dimension if needed: (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
        if len(rgb_features.shape) == 2:
            rgb_features = rgb_features.unsqueeze(1)
        if len(query_features.shape) == 2:
            query_features = query_features.unsqueeze(1)
            
        # Apply MoE processing to RGB features
        enhanced_rgb, aux_loss_rgb, gate_info_rgb = self.moe_module(rgb_features)
        
        # Apply MoE processing to query features
        enhanced_query, aux_loss_query, gate_info_query = self.moe_module(query_features)
        
        # Remove sequence dimension: (batch_size, 1, feature_dim) -> (batch_size, feature_dim)
        enhanced_rgb = enhanced_rgb.squeeze(1)
        enhanced_query = enhanced_query.squeeze(1)
        
        # Combine auxiliary losses
        total_aux_loss = (aux_loss_rgb + aux_loss_query) * self.aux_loss_weight
        
        # Combine gate information from both modalities
        combined_gate_info = {
            'rgb_gate_info': gate_info_rgb,
            'query_gate_info': gate_info_query
        }
                
        return enhanced_rgb, enhanced_query, total_aux_loss, combined_gate_info