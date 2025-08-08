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

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(name = args.pretrain_choice, image_size = args.img_size, stride_size = args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.is_safetensors = os.path.splitext(args.pretrain_choice)[1].lstrip('.') == 'safetensors'

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 


        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
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

            if 'itc' in self.current_task:
                multi_modal_contrastive_itc_loss = 0                
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
                        
                        # 2.计算原始qg对比学习损失
                        qg_loss = objectives.compute_itc(i_feats, t_feats_modal, logit_scale) / len(query_feats)
                        # qg_loss = 0.8 * qg_loss

                    if self.autocast_dtype == torch.float16 and scaler is not None:
                        scaler.scale(qg_loss).backward()
                    else:
                        qg_loss.backward()
                    
                    '''
                    # 跨模态对比学习：需要重新计算特征因为计算图已被清空
                    other_modals = [k for k in query_feats.keys() if k != modal_name]
                    if other_modals:
                        with autocast(device_type='cuda', dtype=self.autocast_dtype):
                            # 重新计算当前模态的特征
                            if modal_name == 'text':
                                _, _, _, _, text_feats_new = self.base_model(text=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal_new = text_feats_new.float()
                                else:
                                    t_feats_modal_new = text_feats_new[torch.arange(text_feats_new.shape[0]), modal_data.argmax(dim=-1)].float()
                            elif modal_name == 'cp':
                                _, cp_img_feats_new, _, _, _ = self.base_model(cp_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal_new = cp_img_feats_new.float()
                                else:
                                    t_feats_modal_new = cp_img_feats_new[:,0,:].float()
                            elif modal_name == 'sk':
                                _, _, sk_img_feats_new, _, _ = self.base_model(sk_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal_new = sk_img_feats_new.float()
                                else:
                                    t_feats_modal_new = sk_img_feats_new[:,0,:].float()
                            elif modal_name == 'nir':
                                _, _, _, nir_img_feats_new, _ = self.base_model(nir_images=modal_data)
                                if self.is_safetensors:
                                    t_feats_modal_new = nir_img_feats_new.float()
                                else:
                                    t_feats_modal_new = nir_img_feats_new[:,0,:].float()

                            # 随机选择一个其他模态
                            random_modal = random.choice(other_modals)
                            random_modal_data = query_feats[random_modal]
                            
                            # 计算随机选择模态的特征
                            if random_modal == 'text':
                                _, _, _, _, random_text_feats = self.base_model(text=random_modal_data)
                                if self.is_safetensors:
                                    random_feats = random_text_feats.float()
                                else:
                                    random_feats = random_text_feats[torch.arange(random_text_feats.shape[0]), random_modal_data.argmax(dim=-1)].float()
                            elif random_modal == 'cp':
                                _, random_cp_feats, _, _, _ = self.base_model(cp_images=random_modal_data)
                                if self.is_safetensors:
                                    random_feats = random_cp_feats.float()
                                else:
                                    random_feats = random_cp_feats[:,0,:].float()
                            elif random_modal == 'sk':
                                _, _, random_sk_feats, _, _ = self.base_model(sk_images=random_modal_data)
                                if self.is_safetensors:
                                    random_feats = random_sk_feats.float()
                                else:
                                    random_feats = random_sk_feats[:,0,:].float()
                            elif random_modal == 'nir':
                                _, _, _, random_nir_feats, _ = self.base_model(nir_images=random_modal_data)
                                if self.is_safetensors:
                                    random_feats = random_nir_feats.float()
                                else:
                                    random_feats = random_nir_feats[:,0,:].float()
                            
                            # 重新计算跨模态对比学习损失
                            cross_modal_loss = objectives.compute_itc(t_feats_modal_new, random_feats, logit_scale) / len(query_feats)
                            cross_modal_loss = 0.2 * cross_modal_loss  # 给跨模态损失一个权重
                        
                        # 在autocast外面进行跨模态损失的backward
                        if self.autocast_dtype == torch.float16 and scaler is not None:
                            scaler.scale(cross_modal_loss).backward()
                        else:
                            cross_modal_loss.backward()
                        
                        # 更新合并损失用于记录
                        loss = qg_loss + cross_modal_loss
                    '''
                    
                    ret.update({f'{modal_name}_itc_Loss': qg_loss.detach()}) # detach后不带计算图, 大写L避免被计入总损失        
                    multi_modal_contrastive_itc_loss += qg_loss
                # multi_modal_contrastive_itc_loss.backward()
                ret.update({'multi_modal_contrastive_itc_loss': multi_modal_contrastive_itc_loss.detach()})

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
                i_feats = vis_img_feat
                t_feats = (text_feat + cp_img_feat+sk_img_feat+nir_img_feat) * 0.25
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
                i_feats = vis_img_feat
                t_feats = (text_feat + cp_img_feat+sk_img_feat+nir_img_feat) * 0.25
            
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


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # convert model to fp16 将模型权重转化为fp16
    if os.path.splitext(args.pretrain_choice)[1].lstrip('.') != 'safetensors':
        convert_weights(model)
    return model
