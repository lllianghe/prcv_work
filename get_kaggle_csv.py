from prettytable import PrettyTable
import os
import torch
import numpy as np
import time
import os.path as op

from transformers.models.auto.feature_extraction_auto import get_feature_extractor_config
from utils.metrics import Evaluator, Evaluator_OR
from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
from torch.utils.data import DataLoader
import argparse
from PIL import Image
from utils.iotools import load_train_configs
from torch.utils.data import Dataset
from datasets.build import build_transforms
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
import json
from utils.kaggle import get_query_type_idx_range
from datasets.bases_or import tokenize
import torch.nn.functional as F
import csv

class KaggleInputDataset(Dataset):
    def __init__(self, json_path,begin_idx,end_idx, transform):
        self.begin_idx = begin_idx
        self.end_idx = end_idx
        self.json_path = json_path
        self.transform = transform
        self.tokenizer = SimpleTokenizer()
        self.text_length = 77
        self.truncate = True
        self.data = self.load_data()

    def load_data(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def __getitem__(self, idx):
        nir_img, cp_img, sk_img, caption = None, None, None, None
        query = self.data[idx + begin_idx]
        query_idx = query['query_idx']
        query_type = query['query_type']
        content = query['content']
        nir_image_path, cp_image_path, sk_image_path, text_description = None, None, None, None
        query_modals = query_type.split('_')[1:]
        for i, modal in enumerate(query_modals):
            if modal == 'TEXT' and i < len(content):
                text_description = content[i]
                caption = tokenize(text_description, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
            elif modal == 'NIR' and i < len(content):
                nir_image_path = content[i]
            elif modal == 'CP' and i < len(content):
                cp_image_path = content[i]
            elif modal == 'SK' and i < len(content):
                sk_image_path = content[i]
        folder_path = '/SSD_Data01/PRCV-ReID5o/data/ORBench_PRCV/val' 
        nir_image = read_image(os.path.join(folder_path, nir_image_path)) if nir_image_path else None
        cp_image = read_image(os.path.join(folder_path, cp_image_path)) if cp_image_path else None
        sk_image = read_image(os.path.join(folder_path, sk_image_path)) if sk_image_path else None
        # 如果文本为空 则将文本设为无描述
        if not text_description:
            caption = tokenize('no description', tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        
        # 如果图像为空，使用空白图像
        nir_img = Image.new("RGB", (224, 224), (0, 0, 0)) if nir_image is None else nir_image
        cp_img = Image.new("RGB", (224, 224), (0, 0, 0)) if cp_image is None else cp_image
        sk_img = Image.new("RGB", (224, 224), (0, 0, 0)) if sk_image is None else sk_image
        if self.transform:
            nir_img = self.transform(nir_img) if nir_img is not None else nir_img
            cp_img = self.transform(cp_img) if cp_img is not None else cp_img
            sk_img = self.transform(sk_img) if sk_img is not None else sk_img
        
        return query_idx, query_type, cp_img, sk_img, nir_img, caption
    
    def __len__(self):
        return end_idx - begin_idx + 1

class GalleryDataset(Dataset):
    def __init__(self, img_folder, transform):
        self.img_folder = img_folder
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, f"{idx+1}.jpg")
        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)
        return idx, img

    def __len__(self):
        return len(self.img_files)
def embedding_qfeats(model, query_loader,modalities):
    model = model.eval()
    device = next(model.parameters()).device
    modalities_list = modalities.split("_") 

    qids, gids, qfeats, gfeats = [], [], [], []
    for query_idx, query_type, cp_img, sk_img, nir_img, caption in query_loader:
        with torch.no_grad():
            feats = []
            if 'TEXT' in modalities_list:
                caption = caption.to(device)
                text_feat = model.encode_text(caption)
                feats.append(text_feat)

            if 'CP' in modalities_list:
                cp_img = cp_img.to(device)
                cp_feat = model.encode_image(cp_img,'cp')
                feats.append(cp_feat)

            if 'SK' in modalities_list:
                sk_img = sk_img.to(device)
                sk_feat = model.encode_image(sk_img,'sk')
                feats.append(sk_feat)

            if 'NIR' in modalities_list:
                nir_img = nir_img.to(device)
                nir_feat = model.encode_image(nir_img,'nir')
                feats.append(nir_feat)
        img_feats = sum(feats) / len(feats) if feats else None  # 避免空列表除以0的错误
        qfeats.append(img_feats)  
    qfeats = torch.cat(qfeats, 0)
    return qfeats

def embedding_gfeats(model, test_gallery_loader):
    model = model.eval()
    device = next(model.parameters()).device
    gfeats = []
    for idx, img in test_gallery_loader:
        with torch.no_grad():
            img = img.to(device)
            vis_feat = model.encode_image(img, modality='vis')
            gfeats.append(vis_feat)
    gfeats = torch.cat(gfeats, 0)
    return gfeats

def embedding_gfeats_with_multiembeddings(model, test_gallery_loader, use_multimodal_layers_in_pairs):
    model = model.eval()
    device = next(model.parameters()).device
    
    gfeats_dict = {'TEXT': [], 'CP': [], 'SK': [], 'NIR': [], 'VIS': []}
    for idx, img in test_gallery_loader:
        with torch.no_grad():
            if use_multimodal_layers_in_pairs:
                # TEXT
                img = img.to(device)
                text_feat = model.encode_image(img, modality='vis')
                gfeats_dict['TEXT'].append(text_feat)
                # CP
                cp_feat = model.encode_image(img, modality='cp')
                gfeats_dict['CP'].append(cp_feat)
                # SK
                sk_feat = model.encode_image(img, modality='sk')
                gfeats_dict['SK'].append(sk_feat)
                # NIR
                nir_feat = model.encode_image(img, modality='nir')
                gfeats_dict['NIR'].append(nir_feat)
            else:
                img = img.to(device)
                text_feat = model.encode_image(img, modality='vis')
                gfeats_dict['VIS'].append(text_feat)

    for modality in gfeats_dict:
        if gfeats_dict[modality] and len(gfeats_dict[modality]) > 0:
                gfeats_dict[modality] = torch.cat(gfeats_dict[modality], 0)
    return gfeats_dict
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="irra Test")
    # 把对应model的file放这就行了
    parser.add_argument("--config_file", default=
                              '/SSD_Data01/myf/research/PRCV/fgclip_model/prcv_work/logs/ORBench/20250728_051944_large_fgclip/configs.yaml'
                    ) 
    # parser.add_argument("--config_file", default='logs/ORBench/20250715_021439_irra/configs.yaml') #这是fgclip的模型
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.training = False
    args.test_batch_size =256
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    json_file = '/SSD_Data01/PRCV-ReID5o/data/ORBench_PRCV/val/val_queries.json'
    query_type_ranges = get_query_type_idx_range(json_file)
    test_transforms = build_transforms(img_size=args.img_size,is_train=False)
    test_gallery_dataset = GalleryDataset('/SSD_Data01/PRCV-ReID5o/data/ORBench_PRCV/val/gallery',transform=test_transforms)
    test_gallery_loader = DataLoader(test_gallery_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    model = build_model(args,num_classes=int(400*(1-args.test_size))) #num_class必须和之前构建的model中的num_class对应
    
    # 根据参数分别控制multimodal embedding和projection层的初始化
    add_embeddings = args.add_multimodal_embeddings or args.add_multimodal_layers
    add_projections = args.add_multimodal_projections or args.add_multimodal_layers
    
    if add_embeddings:
        # 初始化多模态embedding层
        if hasattr(model.base_model, 'vision_model'):
            modalities = ['vis', 'sk', 'nir', 'cp']
            for modality in modalities:
                model.base_model.vision_model.embeddings.add_patch_embedding(modality)
    
    if add_projections:
        # 初始化多模态projection层
        model.base_model.setup_multi_projections()
    
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    
    # 判断是否使用多模态特征提取（需要同时有embedding和projection）
    use_multimodal = add_embeddings and add_projections
    
    # 统一使用 embedding_gfeats_with_multiembeddings，它会根据 use_multimodal_layers_in_pairs 参数处理回退逻辑
    gfeats_dict = embedding_gfeats_with_multiembeddings(model, test_gallery_loader, args.use_multimodal_layers_in_pairs if use_multimodal else False)
    print(f"embedding_gfeats_with_multiembeddings success")
    json_file = '/SSD_Data01/PRCV-ReID5o/data/ORBench_PRCV/val/val_queries.json'
    query_type_ranges = get_query_type_idx_range(json_file)
    output_file=op.join(args.output_dir, 'ranking_list.csv')
    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = ['query_idx', 'query_type', 'ranking_list_idx']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for current_query_type, begin_idx, end_idx in query_type_ranges:
            test_query_dataset = KaggleInputDataset('/SSD_Data01/PRCV-ReID5o/data/ORBench_PRCV/val/val_queries.json', begin_idx, end_idx, test_transforms)
            test_query_loader = DataLoader(test_query_dataset, batch_size=args.test_batch_size, shuffle=False)
            qfeats = embedding_qfeats(model, test_query_loader, current_query_type)
            modalities_list = current_query_type.split("_") 
            
            # 根据 use_multimodal 和 use_multimodal_layers_in_pairs 选择 gallery 特征组合方式
            if use_multimodal and args.use_multimodal_layers_in_pairs:
                # 多模态且使用分支配对：组合对应的模态特征
                gfeats_to_combine = [gfeats_dict[m] for m in modalities_list if m in gfeats_dict and len(gfeats_dict[m]) > 0]
                if not gfeats_to_combine:
                    logger.warning(f"No features found for query type: {current_query_type}. Skipping.")
                    continue
                gfeats = sum(gfeats_to_combine) / len(gfeats_to_combine)
            else:
                # 单模态或多模态但不使用分支配对：使用单一 VIS 特征
                gfeats = gfeats_dict["VIS"]
            qfeats = F.normalize(qfeats, p=2, dim=1)  # 归一化
            gfeats = F.normalize(gfeats, p=2, dim=1)
            similarity = qfeats @ gfeats.t()  # q * g
            indices = torch.argsort(similarity, dim=1, descending=True)  # 排序
            for i, query_idx in enumerate(range(begin_idx, end_idx + 1)):
                ranking_list_idx = (indices[i, :100].cpu().numpy() + 1).tolist()   # 获取前 100 个索引
                writer.writerow({
                    'query_idx': query_idx+1,
                    'query_type': current_query_type,
                    'ranking_list_idx': ranking_list_idx
                })
            print(f"{current_query_type} success")
    
    print("generate csv file success!")
