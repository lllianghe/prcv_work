from torch.nn.modules.loss import _Loss
from model import objectives
from .clip_model import convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import importlib
import os
import gc
import random

from torch.amp import autocast,GradScaler
from .moe_module import PersonReIDMoEProcessor
from .moe_mlp import MoEMLPLayer


# 主模型
class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        # 设置精度控制参数，默认使用全精度32
        self.autocast_dtype = getattr(args, 'autocast_dtype', torch.float32)
        if args.dataset_name == "ORBench":
            from .clip_model_or import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
        else:
            # 原先的导入方法
            from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights

        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(name = args.pretrain_choice, image_size = args.img_size, stride_size = args.stride_size, args=args)
        self.embed_dim = base_cfg['embed_dim']
        self.is_safetensors = os.path.splitext(args.pretrain_choice)[1].lstrip('.') == 'safetensors'

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 


        if 'id' in args.loss_names:
            # For safetensors (FGCLIP/Transformers), encode_image/text return projected features with dim = projection_dim
            # For original OpenAI .pt models, we use the CLS token feature with dim = embed_dim
            id_in_features = self.base_model.config.projection_dim if self.is_safetensors else self.embed_dim
            self.classifier = nn.Linear(id_in_features, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        # 多模态层设置已移至train.py中checkpointer.load之前
        # 这样可以确保新添加的层能够正确地深拷贝预训练权重
        
        # Initialize MoE module for multi-modal feature enhancement with cosine routing
        # MoE模块用于在模型最后一层输出后进行多模态特征增强，使用余弦路由机制
        moe_feature_dim = self.base_model.config.projection_dim if self.is_safetensors else self.embed_dim
        self.moe_processor = PersonReIDMoEProcessor(
            input_dim=moe_feature_dim,
            hidden_dim=moe_feature_dim * 2,  # 专家网络隐藏层维度
            num_experts=args.moe_num_experts,
            top_k=args.moe_top_k,
            aux_loss_weight=args.moe_modal_aux_loss_weight,
        )
        
        # Store global aux loss weight for later use
        self.moe_global_aux_loss_weight = args.moe_global_aux_loss_weight
        
        # MoE MLP层替换配置参数设置
        self.global_aux_loss_weight = getattr(args, 'global_aux_loss_weight', 1.0)
        self.modal_aux_loss_weight = getattr(args, 'modal_aux_loss_weight', 1.0)
        self.moe_mlp_layers = getattr(args, 'moe_mlp_layers', 0)
        self.moe_mlp_lr = getattr(args, 'moe_mlp_lr', None)
        
        # MoE MLP层替换功能集成
        if self.moe_mlp_layers > 0:
            # 替换vision encoder的后6层MLP为MoE MLP
            layer_indices = list(range(6, 12))  # 索引6-11
            replace_vision_mlp_with_moe(
                self.base_model, 
                layer_indices=layer_indices,
                num_experts=8,
                top_k=2
            )
            print(f"已将vision encoder的第{layer_indices}层MLP替换为MoE MLP，专家数量：8，top-k：2")
        
    def collect_moe_mlp_aux_loss(self):
        """收集MoE MLP层的辅助损失和门控信息"""
        total_aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        gate_info_dict = {}
        
        if self.moe_mlp_layers > 0:
            vision_encoder = self.base_model.vision_model.encoder
            layer_indices = list(range(6, 12))  # 后6层
            
            for layer_idx in layer_indices:
                if layer_idx < len(vision_encoder.layers):
                    layer = vision_encoder.layers[layer_idx]
                    if isinstance(layer.mlp, MoEMLPLayer):
                        # 获取该层的门控信息
                        gate_info = layer.mlp.get_gate_info()
                        if gate_info and 'aux_loss' in gate_info:
                            total_aux_loss = total_aux_loss + gate_info['aux_loss']
                            gate_info_dict[f'layer_{layer_idx}'] = gate_info
        
        return total_aux_loss, gate_info_dict
    
    def reset_moe_mlp_gate_info(self):
        """重置MoE MLP层的门控信息"""
        if self.moe_mlp_layers > 0:
            vision_encoder = self.base_model.vision_model.encoder
            layer_indices = list(range(6, 12))  # 后6层
            
            for layer_idx in layer_indices:
                if layer_idx < len(vision_encoder.layers):
                    layer = vision_encoder.layers[layer_idx]
                    if isinstance(layer.mlp, MoEMLPLayer):
                        layer.mlp.reset_gate_info()
    
    def compute_moe_mlp_global_aux_loss(self, accumulated_features):
        """计算MoE MLP的global aux loss，每层分别计算并直接backward"""
        if self.moe_mlp_layers <= 0 or not accumulated_features:
            return
        
        vision_encoder = self.base_model.vision_model.encoder
        layer_indices = list(range(6, 12))  # 后6层
        detach_feature = self.moe_mlp_layers > 0
        
        # 准备所有模态的特征用于重新路由计算
        all_rgb_feats = []
        all_query_feats = []
        
        for feat_dict in accumulated_features:
            all_rgb_feats.append(feat_dict['rgb_feats'])
            all_query_feats.append(feat_dict['query_feats'])
        
        # 拼接所有模态特征
        if all_rgb_feats and all_query_feats:
            combined_rgb_feats = torch.cat(all_rgb_feats, dim=0)  # [total_batch, dim]
            combined_query_feats = torch.cat(all_query_feats, dim=0)  # [total_batch, dim]
            
            # 对每一层MLP分别计算global aux loss
            for layer_idx in layer_indices:
                if layer_idx < len(vision_encoder.layers):
                    layer = vision_encoder.layers[layer_idx]
                    if isinstance(layer.mlp, MoEMLPLayer):
                        # 使用detached特征重新进行余弦路由计算
                        if detach_feature:
                            rgb_input = combined_rgb_feats.detach()
                            query_input = combined_query_feats.detach()
                        else:
                            rgb_input = combined_rgb_feats
                            query_input = combined_query_feats
                        
                        # 对RGB特征进行MoE路由计算
                        _ = layer.mlp(rgb_input, detach_feature=detach_feature)
                        rgb_gate_info = layer.mlp.get_gate_info()
                        
                        # 对query特征进行MoE路由计算
                        _ = layer.mlp(query_input, detach_feature=detach_feature)
                        query_gate_info = layer.mlp.get_gate_info()
                        
                        # 计算该层的global aux loss（使用所有模态prob均值）
                        if rgb_gate_info and query_gate_info:
                            if 'probs' in rgb_gate_info and 'probs' in query_gate_info:
                                # 计算所有模态概率的均值
                                rgb_probs = rgb_gate_info['probs']  # [batch, num_experts]
                                query_probs = query_gate_info['probs']  # [batch, num_experts]
                                
                                # 合并所有模态的概率分布
                                all_probs = torch.cat([rgb_probs, query_probs], dim=0)  # [2*batch, num_experts]
                                mean_probs = torch.mean(all_probs, dim=0)  # [num_experts]
                                
                                # 计算load balancing loss
                                num_experts = mean_probs.size(0)
                                expected_load = 1.0 / num_experts
                                load_loss = torch.sum((mean_probs - expected_load) ** 2)
                                
                                # 应用权重并直接backward
                                weighted_loss = load_loss * self.global_aux_loss_weight
                                if weighted_loss.requires_grad:
                                    weighted_loss.backward(retain_graph=True)
                                
                                print(f"Layer {layer_idx} MLP Global Aux Loss: {weighted_loss.item():.6f}")
                        
                        # 重置gate info为下一次计算准备
                        layer.mlp.reset_gate_info()

    def _set_task(self): # 打印任务
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image, modality=''):
        x = self.base_model.encode_image(image, modality)
        if self.is_safetensors:
            return x.float()
        else:
            return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        if self.is_safetensors:
            return x
        else:
            return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch, scaler=None):
        ret = dict()
        if 'multi_modal_contrastive' in self.current_task: # 新增：多模态对比损失
            vis_images = batch['vis_images']
            cp_images = batch['cp_images']
            sk_images = batch['sk_images']
            nir_images = batch['nir_images']
            caption_ids = batch['caption_ids']
            query_feats = {
                'text': caption_ids,
                'cp': cp_images,
                'sk': sk_images,
                'nir': nir_images
            }
            logit_scale = self.logit_scale
            ret.update({'temperature': 1 / logit_scale})
            if self.args.use_multimodal_layers_in_pairs:
                if 'itc' in self.current_task:
                    multi_modal_contrastive_itc_loss = 0
                    # Initialize variables for global auxiliary loss computation
                    accumulated_features = []  # Store detached features for global aux loss
                    total_tokens = 0
                    
                    for modal_name, modal_data in query_feats.items(): # 遍历每个查询模态
                        with autocast(device_type='cuda', dtype=self.autocast_dtype):
                            # 1.encoder计算特征
                            if modal_name == 'text':
                                vis_img_feats, _, _, _, _ = self.base_model(vis_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, _, _, _, text_feats = self.base_model(text=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = text_feats.float()
                                else:
                                    t_feats_modal = text_feats[torch.arange(text_feats.shape[0]), modal_data.argmax(dim=-1)].float()

                            elif modal_name == 'cp':
                                _, vis_img_feats, _, _, _ = self.base_model(cp_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, cp_img_feats, _, _, _ = self.base_model(cp_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = cp_img_feats.float()
                                else:
                                    t_feats_modal = cp_img_feats[:,0,:].float()

                            elif modal_name == 'sk':
                                _, _, vis_img_feats, _, _ = self.base_model(sk_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, _, sk_img_feats, _, _ = self.base_model(sk_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = sk_img_feats.float()
                                else:
                                    t_feats_modal = sk_img_feats[:,0,:].float()

                            elif modal_name == 'nir':
                                _, _, _, vis_img_feats, _ = self.base_model(nir_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, _, _, nir_img_feats, _ = self.base_model(nir_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = nir_img_feats.float()
                                else:
                                    t_feats_modal = nir_img_feats[:,0,:].float()
                            
                            # Save detached features for global auxiliary loss computation
                            # 保存detach的特征用于全局辅助损失计算
                            accumulated_features.append({
                                'rgb_feats': i_feats.detach(),
                                'query_feats': t_feats_modal.detach()
                            })
                            
                            # Apply MoE processing to rgb and current query modal features
                            # 对rgb和当前query模态特征应用MoE处理（使用余弦路由）
                            i_feats, t_feats_modal, moe_aux_loss, gate_info = self.moe_processor(
                                i_feats, t_feats_modal
                            )
                            
                            # Store gate information for logging
                            ret.update({f'{modal_name}_gate_info': gate_info})
                            
                            # 2.计算原始qg对比学习损失
                            qg_loss = objectives.compute_itc(i_feats, t_feats_modal, logit_scale) / len(query_feats)
                            
                            # 收集MoE MLP层的辅助损失
                            moe_mlp_aux_loss, moe_mlp_gate_info = self.collect_moe_mlp_aux_loss()
                            
                            # Add per-modal MoE auxiliary loss with reduced weight
                            ret.update({f'{modal_name}_itc_loss': qg_loss.detach()}) # detach后不带计算图, 小写l参与总损失计算        
                            ret.update({f'{modal_name}_itc_aux_loss': moe_aux_loss.detach()}) # detach后不带计算图, 小写l参与总损失计算        
                            ret.update({f'{modal_name}_moe_mlp_aux_loss': moe_mlp_aux_loss.detach()}) # MoE MLP辅助损失
                            ret.update({f'{modal_name}_moe_mlp_gate_info': moe_mlp_gate_info}) # MoE MLP门控信息
                            
                            # 将MoE Adapter和MoE MLP的辅助损失都加入主损失
                            qg_loss = qg_loss + moe_aux_loss + moe_mlp_aux_loss * self.modal_aux_loss_weight
                            # qg_loss = 0.8 * qg_loss

                        if self.autocast_dtype == torch.float16 and scaler is not None:
                            scaler.scale(qg_loss).backward()
                        else:
                            qg_loss.backward()
                        
                        # 计算MoE MLP的global aux loss（每层分别计算并直接backward）
                        self.compute_moe_mlp_global_aux_loss(accumulated_features)
                        
                        # 重置MoE MLP门控信息
                        self.reset_moe_mlp_gate_info()

                        multi_modal_contrastive_itc_loss += qg_loss
                    
                    # Compute global auxiliary loss after processing all modalities
                    # 在处理完所有模态后计算全局辅助损失
                    if accumulated_features:
                        with autocast(device_type='cuda', dtype=self.autocast_dtype):
                            # Re-compute routing for all accumulated features to get gate probabilities
                            # 重新计算所有累积特征的路由以获取门控概率
                            all_gate_probs = []
                            for feat_pair in accumulated_features:
                                # Use MoE processor's routing function to get gate probabilities
                                # 使用MoE处理器的路由函数获取门控概率
                                _, _, _, gate_info = self.moe_processor(
                                    feat_pair['rgb_feats'], feat_pair['query_feats']
                                )
                                # Collect gate probabilities from both rgb and query modalities
                                if 'rgb_gate_info' in gate_info and 'gate_probs_mean' in gate_info['rgb_gate_info']:
                                    all_gate_probs.append(gate_info['rgb_gate_info']['gate_probs_mean'])
                                if 'query_gate_info' in gate_info and 'gate_probs_mean' in gate_info['query_gate_info']:
                                    all_gate_probs.append(gate_info['query_gate_info']['gate_probs_mean'])
                            
                            if all_gate_probs:
                                # Average gate probabilities across all modalities
                                # 对所有模态的门控概率进行平均
                                avg_gate_probs = torch.stack(all_gate_probs).mean(dim=0)
                                
                                # Compute global auxiliary loss using L2 loss
                                # 使用L2损失计算全局辅助损失
                                num_experts = avg_gate_probs.size(0)
                                uniform_prob = 1.0 / num_experts
                                expert_fractions = avg_gate_probs / avg_gate_probs.sum()
                                global_aux_loss = torch.sum((expert_fractions - uniform_prob) ** 2)
                                global_aux_loss = global_aux_loss * self.moe_global_aux_loss_weight
                                
                                # Add global auxiliary loss to the total loss
                                # 将全局辅助损失加入总损失
                                ret.update({'global_aux_loss': global_aux_loss.detach()})
                        
                        # Backward pass for global auxiliary loss (outside autocast)
                        if all_gate_probs:
                            if self.autocast_dtype == torch.float16 and scaler is not None:
                                scaler.scale(global_aux_loss).backward()
                            else:
                                global_aux_loss.backward()
                    
                    # multi_modal_contrastive_itc_loss.backward()
                    ret.update({'multi_modal_contrastive_itc_Loss': multi_modal_contrastive_itc_loss.detach()})

                if 'sdm' in self.current_task:
                    multi_modal_contrastive_sdm_loss = 0
                    for modal_name, modal_data in query_feats.items(): # 遍历每个查询模态
                        with autocast(device_type='cuda', dtype=self.autocast_dtype):

                            # 1.encoder计算特征
                            if modal_name == 'text':
                                vis_img_feats, _, _, _, _ = self.base_model(vis_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, _, _, _, text_feats = self.base_model(text=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = text_feats.float()
                                else:
                                    t_feats_modal = text_feats[torch.arange(text_feats.shape[0]), modal_data.argmax(dim=-1)].float()

                            elif modal_name == 'cp':
                                _, vis_img_feats, _, _, _ = self.base_model(cp_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, cp_img_feats, _, _, _ = self.base_model(cp_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = cp_img_feats.float()
                                else:
                                    t_feats_modal = cp_img_feats[:,0,:].float()

                            elif modal_name == 'sk':
                                _, _, vis_img_feats, _, _ = self.base_model(sk_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, _, sk_img_feats, _, _ = self.base_model(sk_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = sk_img_feats.float()
                                else:
                                    t_feats_modal = sk_img_feats[:,0,:].float()

                            elif modal_name == 'nir':
                                _, _, _, vis_img_feats, _ = self.base_model(nir_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, _, _, nir_img_feats, _ = self.base_model(nir_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = nir_img_feats.float()
                                else:
                                    t_feats_modal = nir_img_feats[:,0,:].float()

                            # Apply MoE processing to rgb and current query modal features
                            # 对rgb和当前query模态特征应用MoE处理（使用余弦路由）
                            i_feats, t_feats_modal, moe_aux_loss, gate_info = self.moe_processor(
                                i_feats, t_feats_modal
                            )
                            
                            # 收集MoE MLP层的辅助损失
                            moe_mlp_aux_loss, moe_mlp_gate_info = self.collect_moe_mlp_aux_loss()
                            
                            # Store gate information for logging
                            ret.update({f'{modal_name}_gate_info': gate_info})
                            ret.update({f'{modal_name}_moe_mlp_gate_info': moe_mlp_gate_info})

                            # 2.计算loss
                            loss = objectives.compute_sdm(i_feats, t_feats_modal, batch['pids'], logit_scale) / len(query_feats)
                            # Add MoE Adapter and MoE MLP auxiliary loss
                            loss = loss + moe_aux_loss + moe_mlp_aux_loss * self.modal_aux_loss_weight
                        ret.update({f'{modal_name}_sdm_Loss': loss.detach()}) # .detach()后不带计算图, 大写L避免被计入总损失
                        # 根据精度类型决定是否使用scaler
                        if self.autocast_dtype == torch.float16 and scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        
                        # 重置MoE MLP门控信息
                        self.reset_moe_mlp_gate_info()
                        
                        multi_modal_contrastive_sdm_loss += loss
                    # multi_modal_contrastive_sdm_loss.backward()
                    ret.update({'multi_modal_contrastive_sdm_loss': multi_modal_contrastive_sdm_loss.detach()})
            else :
                if 'itc' in self.current_task:
                    multi_modal_contrastive_itc_loss = 0
                    # Initialize variables for global auxiliary loss computation
                    accumulated_features = []  # Store detached features for global aux loss
                    total_tokens = 0
                    
                    for modal_name, modal_data in query_feats.items(): # 遍历每个查询模态
                        with autocast(device_type='cuda', dtype=self.autocast_dtype):
                            # 1.encoder计算特征
                            if modal_name == 'text':
                                vis_img_feats, _, _, _, _ = self.base_model(vis_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, _, _, _, text_feats = self.base_model(text=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = text_feats.float()
                                else:
                                    t_feats_modal = text_feats[torch.arange(text_feats.shape[0]), modal_data.argmax(dim=-1)].float()

                            elif modal_name == 'cp':
                                vis_img_feats, _, _, _, _ = self.base_model(vis_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, cp_img_feats, _, _, _ = self.base_model(cp_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = cp_img_feats.float()
                                else:
                                    t_feats_modal = cp_img_feats[:,0,:].float()

                            elif modal_name == 'sk':
                                vis_img_feats, _, _, _, _ = self.base_model(vis_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, _, sk_img_feats, _, _ = self.base_model(sk_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = sk_img_feats.float()
                                else:
                                    t_feats_modal = sk_img_feats[:,0,:].float()

                            elif modal_name == 'nir':
                                vis_img_feats, _, _, _, _ = self.base_model(vis_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, _, _, nir_img_feats, _ = self.base_model(nir_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = nir_img_feats.float()
                                else:
                                    t_feats_modal = nir_img_feats[:,0,:].float()
                            
                            # Save detached features for global auxiliary loss computation
                            # 保存detach的特征用于全局辅助损失计算
                            accumulated_features.append({
                                'rgb_feats': i_feats.detach(),
                                'query_feats': t_feats_modal.detach()
                            })
                            
                            # Apply MoE processing to rgb and current query modal features
                            # 对rgb和当前query模态特征应用MoE处理（使用余弦路由）
                            i_feats, t_feats_modal, moe_aux_loss, gate_info = self.moe_processor(
                                i_feats, t_feats_modal
                            )
                            
                            ret.update({f'{modal_name}_gate_info': gate_info})
                            
                            # 2.计算原始qg对比学习损失
                            qg_loss = objectives.compute_itc(i_feats, t_feats_modal, logit_scale) / len(query_feats)
                            
                            # 收集MoE MLP层的辅助损失
                            moe_mlp_aux_loss, moe_mlp_gate_info = self.collect_moe_mlp_aux_loss()
                            
                            # Add per-modal MoE auxiliary loss with reduced weight
                            ret.update({f'{modal_name}_itc_loss': qg_loss.detach()}) # detach后不带计算图, 小写l参与总损失计算        
                            ret.update({f'{modal_name}_itc_aux_loss': moe_aux_loss.detach()}) # detach后不带计算图, 小写l参与总损失计算        
                            ret.update({f'{modal_name}_moe_mlp_aux_loss': moe_mlp_aux_loss.detach()}) # MoE MLP辅助损失
                            ret.update({f'{modal_name}_moe_mlp_gate_info': moe_mlp_gate_info}) # MoE MLP门控信息
                            
                            # 将MoE Adapter和MoE MLP的辅助损失都加入主损失
                            qg_loss = qg_loss + moe_aux_loss + moe_mlp_aux_loss * self.modal_aux_loss_weight
                            # qg_loss = 0.8 * qg_loss

                        if self.autocast_dtype == torch.float16 and scaler is not None:
                            scaler.scale(qg_loss).backward()
                        else:
                            qg_loss.backward()
                        
                        # 计算MoE MLP的global aux loss（每层分别计算并直接backward）
                        self.compute_moe_mlp_global_aux_loss(accumulated_features)
                        
                        # 重置MoE MLP门控信息
                        self.reset_moe_mlp_gate_info()


                        multi_modal_contrastive_itc_loss += qg_loss
                    
                    # Compute global auxiliary loss after processing all modalities
                    # 在处理完所有模态后计算全局辅助损失
                    if accumulated_features:
                        with autocast(device_type='cuda', dtype=self.autocast_dtype):
                            # Re-compute routing for all accumulated features to get gate probabilities
                            # 重新计算所有累积特征的路由以获取门控概率
                            all_gate_probs = []
                            for i, feat_pair in enumerate(accumulated_features):
                                # Use MoE processor's routing function to get gate probabilities
                                # 使用MoE处理器的路由函数获取门控概率
                                _, _, _, gate_info = self.moe_processor(
                                    feat_pair['rgb_feats'], feat_pair['query_feats']
                                )
                                # Collect gate probabilities from both rgb and query modalities
                                if 'rgb_gate_info' in gate_info and 'gate_probs_mean' in gate_info['rgb_gate_info']:
                                    all_gate_probs.append(gate_info['rgb_gate_info']['gate_probs_mean'])
                                if 'query_gate_info' in gate_info and 'gate_probs_mean' in gate_info['query_gate_info']:
                                    all_gate_probs.append(gate_info['query_gate_info']['gate_probs_mean'])
                            
                            if all_gate_probs:

                                # Average gate probabilities across all modalities
                                # 对所有模态的门控概率进行平均
                                avg_gate_probs = torch.stack(all_gate_probs).mean(dim=0)
                                
                                # Compute global auxiliary loss using L2 loss
                                # 使用L2损失计算全局辅助损失
                                num_experts = avg_gate_probs.size(0)
                                uniform_prob = 1.0 / num_experts
                                expert_fractions = avg_gate_probs / avg_gate_probs.sum()
                                global_aux_loss = torch.sum((expert_fractions - uniform_prob) ** 2)
                                global_aux_loss = global_aux_loss * self.moe_global_aux_loss_weight
                                
                                # Add global auxiliary loss to the total loss
                                # 将全局辅助损失加入总损失
                                ret.update({'global_aux_loss': global_aux_loss.detach()})
                        
                        # Backward pass for global auxiliary loss (outside autocast)
                        if 'global_aux_loss' in locals():
                            if self.autocast_dtype == torch.float16 and scaler is not None:
                                scaler.scale(global_aux_loss).backward()
                            else:
                                global_aux_loss.backward()
                    
                    # multi_modal_contrastive_itc_loss.backward()
                    ret.update({'multi_modal_contrastive_itc_Loss': multi_modal_contrastive_itc_loss.detach()})

                if 'sdm' in self.current_task:
                    multi_modal_contrastive_sdm_loss = 0
                    for modal_name, modal_data in query_feats.items(): # 遍历每个查询模态
                        with autocast(device_type='cuda', dtype=self.autocast_dtype):

                            # 1.encoder计算特征
                            if modal_name == 'text':
                                vis_img_feats, _, _, _, _ = self.base_model(vis_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, _, _, _, text_feats = self.base_model(text=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = text_feats.float()
                                else:
                                    t_feats_modal = text_feats[torch.arange(text_feats.shape[0]), modal_data.argmax(dim=-1)].float()

                            elif modal_name == 'cp':
                                vis_img_feats, _, _, _, _ = self.base_model(vis_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, cp_img_feats, _, _, _ = self.base_model(cp_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = cp_img_feats.float()
                                else:
                                    t_feats_modal = cp_img_feats[:,0,:].float()

                            elif modal_name == 'sk':
                                vis_img_feats, _, _, _, _ = self.base_model(vis_images=vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, _, sk_img_feats, _, _ = self.base_model(sk_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = sk_img_feats.float()
                                else:
                                    t_feats_modal = sk_img_feats[:,0,:].float()

                            elif modal_name == 'nir':
                                vis_img_feats, _, _, _, _ = self.base_model(vis_images==vis_images)
                                if self.is_safetensors:
                                    i_feats = vis_img_feats.float()
                                else:
                                    i_feats = vis_img_feats[:,0,:].float()
                                _, _, _, nir_img_feats, _ = self.base_model(nir_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal = nir_img_feats.float()
                                else:
                                    t_feats_modal = nir_img_feats[:,0,:].float()

                            # 2.计算loss
                            loss = objectives.compute_sdm(i_feats, t_feats_modal, batch['pids'], logit_scale) / len(query_feats)
                        ret.update({f'{modal_name}_sdm_Loss': loss.detach()}) # .detach()后不带计算图, 大写L避免被计入总损失
                        # 根据精度类型决定是否使用scaler
                        if self.autocast_dtype == torch.float16 and scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        multi_modal_contrastive_sdm_loss += loss
                    # multi_modal_contrastive_sdm_loss.backward()
                    ret.update({'multi_modal_contrastive_sdm_loss': multi_modal_contrastive_sdm_loss.detach()})
  

            # 如果需要计算后续损失, 则重新计算i_feats和t_feats以兼容id loss
            if any(task in self.current_task for task in ['id', 'mlm', 'cmpm']):
                vis_img_feats, cp_img_feats, sk_img_feats, nir_img_feats, text_feats = self.base_model(
                    vis_images, cp_images, sk_images, nir_images, caption_ids
                )
                if self.is_safetensors:
                    vis_img_feat = vis_img_feats.float()
                    cp_img_feat = cp_img_feats.float()
                    sk_img_feat = sk_img_feats.float()
                    nir_img_feat = nir_img_feats.float()
                    text_feat = text_feats.float()
                else:
                    vis_img_feat = vis_img_feats[:,0,:].float()
                    cp_img_feat = cp_img_feats[:,0,:].float()
                    sk_img_feat = sk_img_feats[:,0,:].float()
                    nir_img_feat = nir_img_feats[:,0,:].float()
                    text_feat = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
                
                # Apply MoE processing to multi-modal features (using cosine routing)
                # 对多模态特征应用MoE处理（使用余弦路由）- 两两配对处理
                
                # Process vis-cp pair
                vis_img_feat, cp_img_feat, moe_aux_loss_1 = self.moe_processor(
                    vis_img_feat, cp_img_feat
                )
                
                # Process vis-sk pair (using updated vis features)
                vis_img_feat, sk_img_feat, moe_aux_loss_2 = self.moe_processor(
                    vis_img_feat, sk_img_feat
                )
                
                # Process vis-nir pair (using updated vis features)
                vis_img_feat, nir_img_feat, moe_aux_loss_3 = self.moe_processor(
                    vis_img_feat, nir_img_feat
                )
                
                # Combine auxiliary losses
                moe_aux_loss = (moe_aux_loss_1 + moe_aux_loss_2 + moe_aux_loss_3) / 3.0
                
                i_feats = vis_img_feat
                t_feats = (text_feat + cp_img_feat+sk_img_feat+nir_img_feat) * 0.25
                
                # Add MoE auxiliary loss to return dict
                ret.update({'moe_aux_loss': moe_aux_loss})
        else:
            if self.args.dataset_name == 'ORBench':
                vis_images = batch['vis_images']
                cp_images = batch['cp_images']
                sk_images = batch['sk_images']
                nir_images = batch['nir_images']
                caption_ids = batch['caption_ids']
                vis_img_feats, cp_img_feats, sk_img_feats, nir_img_feats, text_feats = self.base_model(
                    vis_images, cp_images, sk_images, nir_images, caption_ids
                )
                if self.is_safetensors:
                    vis_img_feat = vis_img_feats.float()
                    cp_img_feat = cp_img_feats.float()
                    sk_img_feat = sk_img_feats.float()
                    nir_img_feat = nir_img_feats.float()
                    text_feat = text_feats.float()
                else:
                    vis_img_feat = vis_img_feats[:,0,:].float()
                    cp_img_feat = cp_img_feats[:,0,:].float()
                    sk_img_feat = sk_img_feats[:,0,:].float()
                    nir_img_feat = nir_img_feats[:,0,:].float()
                    text_feat = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
                
                # Apply MoE processing to multi-modal features
                # 对多模态特征应用MoE处理
                features_dict = {
                    'vis': vis_img_feat,
                    'cp': cp_img_feat,
                    'sk': sk_img_feat,
                    'nir': nir_img_feat
                }
                enhanced_features, moe_aux_loss = self.moe_processor(features_dict)
                
                # Update features with MoE enhanced versions
                vis_img_feat = enhanced_features['vis']
                cp_img_feat = enhanced_features['cp']
                sk_img_feat = enhanced_features['sk']
                nir_img_feat = enhanced_features['nir']
                
                i_feats = vis_img_feat
                t_feats = (text_feat + cp_img_feat+sk_img_feat+nir_img_feat) * 0.25
                
                # Add MoE auxiliary loss to return dict
                ret.update({'moe_aux_loss': moe_aux_loss * self.moe_aux_loss_weight})
            
            else:
                images = batch['images']
                caption_ids = batch['caption_ids']
                image_feats, text_feats = self.base_model(images, caption_ids)
                i_feats = image_feats[:, 0, :].float() # 选择第一个特征
                # i_feats = image_feats.float() # for CLIP ResNet visual model
                t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

            logit_scale = self.logit_scale
            ret.update({'temperature': 1 / logit_scale})

            if 'itc' in self.current_task:
                ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
            
            if 'sdm' in self.current_task:
                ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            if self.is_safetensors:
                image_logits = self.classifier(i_feats).float()
                text_logits = self.classifier(t_feats).float()
            else:
                image_logits = self.classifier(i_feats.half()).float()
                text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def copy_mlp_weights_to_moe(original_mlp, moe_mlp, num_experts):
    """将原始CLIPMLP权重复制到所有MoE专家"""
    if moe_mlp.use_moe:
        for i in range(num_experts):
            # 复制fc1权重到每个专家
            moe_mlp.mlp.experts[i].fc1.weight.data.copy_(original_mlp.fc1.weight.data)
            if hasattr(original_mlp.fc1, 'bias') and original_mlp.fc1.bias is not None:
                moe_mlp.mlp.experts[i].fc1.bias.data.copy_(original_mlp.fc1.bias.data)
            # 复制fc2权重到每个专家  
            moe_mlp.mlp.experts[i].fc2.weight.data.copy_(original_mlp.fc2.weight.data)
            if hasattr(original_mlp.fc2, 'bias') and original_mlp.fc2.bias is not None:
                moe_mlp.mlp.experts[i].fc2.bias.data.copy_(original_mlp.fc2.bias.data)


def replace_vision_mlp_with_moe(model, layer_indices=None, num_experts=8, top_k=2, temperature=1.0):
    """将指定层的CLIPMLP替换为MoEMLPLayer"""
    if layer_indices is None or len(layer_indices) == 0:
        return  # 如果没有指定层或为空列表，则不进行替换
    
    # 确保模型有vision_model.encoder.layers结构
    if not (hasattr(model, 'base_model') and 
            hasattr(model.base_model, 'vision_model') and 
            hasattr(model.base_model.vision_model, 'encoder') and 
            hasattr(model.base_model.vision_model.encoder, 'layers')):
        print("Warning: Model structure does not support MoE MLP replacement")
        return
    
    total_layers = len(model.base_model.vision_model.encoder.layers)
    
    for layer_idx in layer_indices:
        if layer_idx < total_layers:
            original_mlp = model.base_model.vision_model.encoder.layers[layer_idx].mlp
            
            # 获取原始MLP的配置
            d_model = original_mlp.config.hidden_size
            d_ff = original_mlp.config.intermediate_size
            
            # 创建MoE MLP层
            moe_mlp = MoEMLPLayer(
                d_model=d_model, 
                d_ff=d_ff, 
                num_experts=num_experts, 
                top_k=top_k, 
                temperature=temperature,
                use_moe=True
            )
            
            # 参数复制逻辑
            copy_mlp_weights_to_moe(original_mlp, moe_mlp, num_experts)
            
            # 替换层
            model.base_model.vision_model.encoder.layers[layer_idx].mlp = moe_mlp
            
            print(f"Replaced layer {layer_idx} MLP with MoE MLP (experts={num_experts}, top_k={top_k})")
        else:
            print(f"Warning: Layer index {layer_idx} exceeds total layers {total_layers}")


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # convert model to fp16 将模型权重转化为fp16
    if os.path.splitext(args.pretrain_choice)[1].lstrip('.') != 'safetensors':
        convert_weights(model)
    return model
