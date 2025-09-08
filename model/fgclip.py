import torch
import torch.nn as nn
import math

from transformers import CLIPConfig,AutoConfig
from typing import Any, Optional, Tuple, Union
import torch.distributed.nn as nn_dist
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Tuple, Union
from .modeling_clip import CLIPModel, CLIPTextTransformer, CLIPVisionTransformer, CLIPOutput, CLIPAttention, CLIPMLP

import torch.distributed as dist
from torch.nn import AvgPool2d
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from torch import nn

import math
from torchvision.ops import roi_align

from model.lnmodal import inject_vision_layernorm

class FGCLIPModel(CLIPModel):
    config_class = CLIPConfig
    main_input_name = "text_long"

    def __init__(self, config):
        super(CLIPModel, self).__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config
        text_config.eos_token_id = 49407
        text_config.pad_token_id = 49407
        text_config.bos_token_id = 49406

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)

        self.vision_model = CLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)


        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.text_filip_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)


        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_finegraind = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_hardneg = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

 
        self.embed_dim = text_config.hidden_size
        self.world_size = 0

        # Initialize weights and apply final processing
        self.post_init()

        '''
        # 插入layernorm层
        '''
        self.lnmodal_modules = inject_vision_layernorm(self.vision_model)
        for lnmodal_module in self.lnmodal_modules:
            self.add_module(lnmodal_module.lnmodal_name, lnmodal_module)

    def resize_postion_embeding(self, newsize=248):

        old_position_embedding = self.text_model.embeddings.position_embedding
        old_position_embedding_res = self.text_model.embeddings.position_embedding_res
        old_position_embedding_ori = self.text_model.embeddings.position_embedding_ori
        
        positional_embedding_pre = self.text_model.embeddings.position_embedding.weight.data
    
        length, dim = positional_embedding_pre.shape
        keep_len = 20
        posisitonal_embedding_new = torch.zeros([4*length-3*keep_len, dim], dtype=positional_embedding_pre.dtype)
        for i in range(keep_len):
            posisitonal_embedding_new[i] = positional_embedding_pre[i]
        for i in range(length-1-keep_len):
            posisitonal_embedding_new[4*i + keep_len] = positional_embedding_pre[i + keep_len]
            posisitonal_embedding_new[4*i + 1 + keep_len] = 3*positional_embedding_pre[i + keep_len]/4 + 1*positional_embedding_pre[i+1+keep_len]/4
            posisitonal_embedding_new[4*i + 2+keep_len] = 2*positional_embedding_pre[i+keep_len]/4 + 2*positional_embedding_pre[i+1+keep_len]/4
            posisitonal_embedding_new[4*i + 3+keep_len] = 1*positional_embedding_pre[i+keep_len]/4 + 3*positional_embedding_pre[i+1+keep_len]/4

        posisitonal_embedding_new[4*length -3*keep_len - 4] = positional_embedding_pre[length-1] + 0*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 3] = positional_embedding_pre[length-1] + 1*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 2] = positional_embedding_pre[length-1] + 2*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 1] = positional_embedding_pre[length-1] + 3*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
                
        positional_embedding_res = posisitonal_embedding_new.clone()

        self.text_model.embeddings.position_embedding_ori.weight.data = posisitonal_embedding_new
        self.text_model.embeddings.position_embedding_ori.num_embeddings = posisitonal_embedding_new.shape[0]
        
        self.text_model.embeddings.position_embedding_res.weight.data = positional_embedding_res
        self.text_model.embeddings.position_embedding_res.num_embeddings = positional_embedding_res.shape[0]

        old_position_embedding_ori_requires_grad = old_position_embedding_ori.weight.requires_grad
        self.text_model.embeddings.position_embedding_ori.requires_grad_(old_position_embedding_ori_requires_grad)

        old_position_embedding_res_requires_grad = old_position_embedding_res.weight.requires_grad
        self.text_model.embeddings.position_embedding_res.requires_grad_(old_position_embedding_res_requires_grad)



    def copy_weight(self,):
        with torch.no_grad():
            self.text_filip_projection.weight.data.copy_(self.text_projection.weight.data)  

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        modality = '',
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        """
        # 注释这几行即可取消ln_modal 切换到对应的模态
        """
        for lnmodal_module in self.lnmodal_modules:
            lnmodal_module.apply_to(modality=modality)

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            modality = modality,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features
    
    def get_image_box_roi_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        box_info=None,
    ) -> torch.FloatTensor:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )

        bs = pixel_values.shape[0]
        length = vision_outputs[0].shape[1]-1
        h = int(math.sqrt(length))
        w = h

        feature_map = vision_outputs.hidden_states[-2]#[:, 1:, :]
        feature_map = self.forward_without_attn(feature_map)[:, 1:]

        feature_map = self.vision_model.post_layernorm(feature_map)
        feature_map = self.visual_projection(feature_map)

        feature_map = feature_map.view(bs, h, w, -1).permute(0, 3, 1, 2)
        x_rois = roi_align(feature_map.type(torch.float32),box_info, (1, 1), 1.0, -1, True)[..., 0, 0]

        # x_rois = x_rois / x_rois.norm(p=2, dim=-1, keepdim=True)

        return x_rois

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        walk_short_pos: Optional[bool] = True,
        use_bbox: Optional[bool] = False
    ) -> torch.FloatTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pos_flag = walk_short_pos or use_bbox

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            walk_short_pos=pos_flag,
        )
        pooled_output = text_outputs[1]

        if walk_short_pos:
            text_features = self.text_projection(pooled_output)
        else:
            text_features = self.text_filip_projection(pooled_output)           

        return text_features

    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        # print("normed_boxes, ", normed_boxes.shape)
        for boxes in normed_boxes:
            # print("boxes, ", boxes)
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes.type(torch.float32))
        return denormed_boxes

    def forward_without_attn(self, x):
        # get last layer 
        residual = x
        x = self.vision_model.encoder.layers[-1].layer_norm1(x)

        x = F.linear(input=x, weight=self.vision_model.encoder.layers[-1].self_attn.v_proj.weight, bias=self.vision_model.encoder.layers[-1].self_attn.v_proj.bias)
        x = self.vision_model.encoder.layers[-1].self_attn.out_proj(x)
        x = residual+x

        residual = x
        x = self.vision_model.encoder.layers[-1].layer_norm2(x)
        x = self.vision_model.encoder.layers[-1].mlp(x)
        x = residual + x

        return x

    def get_image_dense_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        box_info=None,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )


        bs = pixel_values.shape[0]
        length = vision_outputs[0].shape[1]-1
        h = int(math.sqrt(length))
        w = h

        feature_map = vision_outputs.hidden_states[-2]#[:, 1:, :]
        feature_map = self.forward_without_attn(feature_map)[:, 1:]

        feature_map = self.vision_model.post_layernorm(feature_map)
        feature_map = self.visual_projection(feature_map)

        return feature_map


    # def forward( self, vis_images, cp_images, sk_images, nir_images, text):
        vis_img_feats = self.get_image_features(vis_images)
        cp_img_feats = self.get_image_features(cp_images)
        sk_img_feats = self.get_image_features(sk_images)
        nir_img_feats = self.get_image_features(nir_images)
        text_feats = self.get_text_features(text)
        return vis_img_feats, cp_img_feats, sk_img_feats, nir_img_feats, text_feats


    def forward(self, vis_images=None, cp_images=None, sk_images=None, nir_images=None, text=None):
        vis_img_feats = self.encode_image(vis_images, modality = 'vis') if vis_images is not None else None # 小写
        cp_img_feats = self.encode_image(cp_images, modality = 'cp') if cp_images is not None else None
        sk_img_feats = self.encode_image(sk_images, modality = 'sk') if sk_images is not None else None
        nir_img_feats = self.encode_image(nir_images, modality = 'nir') if nir_images is not None else None
        text_feats = self.encode_text(text) if text is not None else None
        return vis_img_feats, cp_img_feats, sk_img_feats, nir_img_feats, text_feats
    
    

    def clip_loss(self,image_features_long, text_features_long, text_features_short,rank,image):

        # (bz, embed_dim)
        image_feat_all_long = torch.cat(nn_dist.all_gather(image_features_long), dim=0)#gather with grad

        if text_features_long is not None:
            text_feat_all_long = torch.cat(nn_dist.all_gather(text_features_long), dim=0)

        text_feat_all_short = torch.cat(nn_dist.all_gather(text_features_short), dim=0)
        
        # the loss function 
        if text_features_long is not None:
            sim_i2tl = torch.matmul(image_features_long, text_feat_all_long.T)
            sim_tl2i = torch.matmul(image_feat_all_long, text_features_long.T)
            sim_tl2i = sim_tl2i.T

        sim_i2ts = torch.matmul(image_features_long, text_feat_all_short.T)
        sim_ts2i = torch.matmul(image_feat_all_long, text_features_short.T)
        sim_ts2i = sim_ts2i.T

        
        if text_features_long is not None:
            sim_i2tl = self.logit_scale.exp() * sim_i2tl
            sim_tl2i = self.logit_scale.exp() * sim_tl2i


        sim_i2ts = self.logit_scale.exp() * sim_i2ts
        sim_ts2i = self.logit_scale.exp() * sim_ts2i

        
        bs = image_features_long.size(0)
        targets = torch.linspace(rank * bs,rank * bs + bs - 1, bs, dtype=torch.long).to(image.device)

        loss_itcl = None
        if text_features_long is not None:
            loss_itcl = (
                    F.cross_entropy(sim_i2tl, targets, label_smoothing=0.0)
                    + F.cross_entropy(sim_tl2i, targets, label_smoothing=0.0)
                ) / 2
        
        loss_itcs = (
                F.cross_entropy(sim_i2ts, targets, label_smoothing=0.0)
                + F.cross_entropy(sim_ts2i, targets, label_smoothing=0.0)
            ) / 2

        return loss_itcl, loss_itcs


    def pairwise_contrastive_loss(self, image_features_long, text_features_long, device, logit_scale=1.0):
        batch_size, c = image_features_long.shape
        labels = torch.eye(batch_size, device=device, dtype=torch.float)#.repeat(batch_size, 1)
        logits_per_image = logit_scale.exp() * image_features_long @ text_features_long.T
        logits_per_text = logit_scale.exp() * text_features_long @ image_features_long.T
        temp1 = F.cross_entropy(logits_per_text, labels)
        temp2 = F.cross_entropy(logits_per_image, labels)

        loss = (temp1+temp2)/2
        return loss
    
    def hard_contrastive_loss(self, image_features_long, text_features_long, device, logit_scale=1.0):
        batch_size, c = image_features_long.shape
        text_features_long = text_features_long.reshape(batch_size, 11, -1)
        labels = torch.zeros(batch_size, device=device, dtype=torch.long)#.repeat(batch_size, 1)
        predict = logit_scale.exp() * torch.einsum('bp,bdp->bd', image_features_long, text_features_long)
        loss = F.cross_entropy(predict, labels)
        return loss
    
    encode_image = get_image_features
    encode_text = get_text_features