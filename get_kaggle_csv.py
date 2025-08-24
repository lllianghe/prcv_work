import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import os.path as op
import csv
from prettytable import PrettyTable
from utils.metrics import Evaluator, Evaluator_OR, rank_with_distance
from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from torch.utils.data import DataLoader
from PIL import Image
from utils.iotools import load_train_configs
from torch.utils.data import Dataset
from datasets.build import build_transforms
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from utils.kaggle import get_query_type_idx_range
from datasets.bases_or import tokenize
from utils.re_ranking import fast_gcrv_image
from utils.cmc import ReRank

class KaggleInputDataset(Dataset):
    def __init__(self, json_path, begin_idx, end_idx, transform):
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
            import json
            data = json.load(f)
        return data
    
    def __getitem__(self, idx):
        nir_img, cp_img, sk_img, caption = None, None, None, None
        query = self.data[idx + self.begin_idx]  # 修正：使用 self.begin_idx
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
        return self.end_idx - self.begin_idx + 1  # 修正：使用 self.end_idx 和 self.begin_idx

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

def embedding_qfeats(model, query_loader, modalities):
    model = model.eval()
    device = next(model.parameters()).device
    modalities_list = modalities.split("_") 

    qfeats = []
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
        img_feats = sum(feats) / len(feats) if feats else None
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

def embedding_gfeats_with_multiembeddings(model, test_gallery_loader):
    model = model.eval()
    device = next(model.parameters()).device
    
    gfeats_dict = {'TEXT': [], 'CP': [], 'SK': [], 'NIR': []}
    for idx, img in test_gallery_loader:
        with torch.no_grad():
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
    for modality in gfeats_dict:
        if gfeats_dict[modality] and len(gfeats_dict[modality]) > 0:
            gfeats_dict[modality] = torch.cat(gfeats_dict[modality], 0)
    return gfeats_dict

def apply_single_reranking(qfeats, gfeats, rerank_method, rerank_cfg=None, logger=None):
    """
    应用单个重排序方法
    """
    try:
        if rerank_method == 'none' or rerank_method is None:
            # 原始相似度排序
            similarity = qfeats @ gfeats.t()
            indices = torch.argsort(similarity, dim=1, descending=True)
            return indices
        
        elif rerank_method == 'rerank':
            # K-reciprocal re-ranking
            k1 = rerank_cfg.get('k1', 20) if rerank_cfg else 20
            k2 = rerank_cfg.get('k2', 6) if rerank_cfg else 6
            lambda_value = rerank_cfg.get('lambda_value', 0.3) if rerank_cfg else 0.3
            
            if logger:
                logger.info(f"Applying k-reciprocal re-ranking with k1={k1}, k2={k2}, lambda={lambda_value}")
            
            dist_matrix = ReRank(qfeats.cpu().numpy(), gfeats.cpu().numpy(), 
                               k1=k1, k2=k2, lambda_value=lambda_value)
            indices = torch.from_numpy(np.argsort(dist_matrix, axis=1))
            return indices
        
        elif rerank_method == 'gcr':
            # GCR re-ranking
            if rerank_cfg is None:
                if logger:
                    logger.error("GCR configuration is required for GCR re-ranking")
                return None
            
            if logger:
                logger.info("Applying GCR re-ranking")
            
            # 准备GCR需要的数据格式
            qfeats_np = qfeats.cpu()
            gfeats_np = gfeats.cpu()
            
            # 创建伪ID和tracks（竞赛中不需要真实的ID）
            qids_np = np.arange(qfeats.shape[0])
            gids_np = np.arange(gfeats.shape[0])
            qids_tracks = np.zeros_like(qids_np)
            gids_tracks = np.zeros_like(gids_np)
            
            all_data = [qfeats_np, qids_np, qids_tracks, gfeats_np, gids_np, gids_tracks]
            dist_matrix, _, _ = fast_gcrv_image(rerank_cfg, all_data)
            indices = torch.from_numpy(np.argsort(dist_matrix, axis=1))
            return indices
        
        else:
            if logger:
                logger.error(f"Unknown rerank method: {rerank_method}")
            return None
            
    except Exception as e:
        if logger:
            logger.error(f"Error in {rerank_method} re-ranking: {str(e)}")
        return None

def apply_reranking(qfeats, gfeats, rerank_methods, rerank_cfg=None, logger=None):
    """
    应用重排序方法，支持多种方法
    """
    if isinstance(rerank_methods, str):
        rerank_methods = [rerank_methods]
    
    results = {}
    for method in rerank_methods:
        if logger:
            logger.info(f"Processing with method: {method}")
        
        # 为GCR设置特定的配置
        if method == 'gcr':
            gcr_cfg = rerank_cfg.get('gcr_cfg') if rerank_cfg else None
        else:
            gcr_cfg = rerank_cfg
        
        indices = apply_single_reranking(qfeats, gfeats, method, gcr_cfg, logger)
        if indices is not None:
            results[method] = indices
        else:
            if logger:
                logger.error(f"Failed to apply {method} re-ranking")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="irra Test")
    # 配置文件路径
    parser.add_argument("--config_file", default=
                              '/SSD_Data01/myf/research/PRCV/fgclip_model/prcv_work/logs/ORBench/20250728_051944_large_fgclip/configs.yaml'
                    ) 
    # 重排序方法选择
    parser.add_argument("--rerank_method", type=str, default=None, 
                       choices=[None, 'none', 'rerank', 'gcr', 'all'],
                       help="Re-ranking method: None/none (original), 'rerank' (k-reciprocal), 'gcr', 'all' (run all methods)")
    
    # K-reciprocal 重排序参数
    parser.add_argument("--rerank_k1", type=int, default=20,
                       help="k1 parameter for k-reciprocal re-ranking")
    parser.add_argument("--rerank_k2", type=int, default=6,
                       help="k2 parameter for k-reciprocal re-ranking")
    parser.add_argument("--rerank_lambda", type=float, default=0.3,
                       help="lambda parameter for k-reciprocal re-ranking")
    
    # GCR 重排序参数 - 根据 re_ranking.py 中的需求补全
    parser.add_argument("--gcr_enable", action='store_true', default=True,
                       help="Enable GCR")
    parser.add_argument("--gcr_gal_round", type=int, default=3,
                       help="Gallery round for GCR (迭代轮数)")
    parser.add_argument("--gcr_beta1", type=float, default=0.1,
                       help="Beta1 parameter for GCR (相似度计算参数)")
    parser.add_argument("--gcr_beta2", type=float, default=0.1,
                       help="Beta2 parameter for GCR (相似度计算参数)")
    parser.add_argument("--gcr_lambda1", type=float, default=2.0,
                       help="Lambda1 parameter for GCR (阈值计算参数)")
    parser.add_argument("--gcr_lambda2", type=float, default=2.0,
                       help="Lambda2 parameter for GCR (阈值计算参数)")
    parser.add_argument("--gcr_scale", type=float, default=1.0,
                       help="Scale parameter for GCR (缩放参数)")
    parser.add_argument("--gcr_mode", type=str, default='sym',
                       choices=['sym', 'fixA', 'no-norm'],
                       help="GCR mode (对称模式/固定A模式/无归一化模式)")
    parser.add_argument("--gcr_with_gpu", action='store_true', default=True,
                       help="Use GPU for GCR")
    parser.add_argument("--gcr_verbose", action='store_true', default=False,
                       help="Verbose output for GCR")
    
    # 先解析命令行参数
    cmd_args = parser.parse_args()
    
    # 保存重排序相关的命令行参数
    rerank_method = cmd_args.rerank_method
    rerank_k1 = cmd_args.rerank_k1
    rerank_k2 = cmd_args.rerank_k2
    rerank_lambda = cmd_args.rerank_lambda
    
    # GCR 参数
    gcr_enable = cmd_args.gcr_enable
    gcr_gal_round = cmd_args.gcr_gal_round
    gcr_beta1 = cmd_args.gcr_beta1
    gcr_beta2 = cmd_args.gcr_beta2
    gcr_lambda1 = cmd_args.gcr_lambda1
    gcr_lambda2 = cmd_args.gcr_lambda2
    gcr_scale = cmd_args.gcr_scale
    gcr_mode = cmd_args.gcr_mode
    gcr_with_gpu = cmd_args.gcr_with_gpu
    gcr_verbose = cmd_args.gcr_verbose
    
    # 加载配置文件
    args = load_train_configs(cmd_args.config_file)
    args.training = False
    args.test_batch_size = 256
    
    # 恢复重排序参数
    args.rerank_method = rerank_method
    args.rerank_k1 = rerank_k1
    args.rerank_k2 = rerank_k2
    args.rerank_lambda = rerank_lambda
    
    # 恢复 GCR 参数
    args.gcr_enable = gcr_enable
    args.gcr_gal_round = gcr_gal_round
    args.gcr_beta1 = gcr_beta1
    args.gcr_beta2 = gcr_beta2
    args.gcr_lambda1 = gcr_lambda1
    args.gcr_lambda2 = gcr_lambda2
    args.gcr_scale = gcr_scale
    args.gcr_mode = gcr_mode
    args.gcr_with_gpu = gcr_with_gpu
    args.gcr_verbose = gcr_verbose
    
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    logger.info(f"Rerank method from command line: {rerank_method}")
    
    # 确定要运行的重排序方法
    if args.rerank_method == 'all':
        rerank_methods = ['rerank', 'gcr','none']
        logger.info("Running all re-ranking methods")
    elif args.rerank_method in ['rerank', 'gcr']:
        rerank_methods = [args.rerank_method]
        logger.info(f"Running single re-ranking method: {args.rerank_method}")
    else:
        rerank_methods = ['none']
        logger.info("Running original similarity ranking")

    logger.info(f"Final rerank_method: {args.rerank_method}")
    logger.info(f"Final rerank_methods list: {rerank_methods}")
    
    # 设置重排序配置
    rerank_cfg = {
        'k1': getattr(args, 'rerank_k1', 20),
        'k2': getattr(args, 'rerank_k2', 6),
        'lambda_value': getattr(args, 'rerank_lambda', 0.3),
        'gcr_cfg': None
    }
    
    # 设置GCR配置 - 包含所有必需参数
    if 'gcr' in rerank_methods:
        from types import SimpleNamespace
        gcr_cfg = SimpleNamespace()
        gcr_cfg.GCR = SimpleNamespace()
        gcr_cfg.GCR.ENABLE_GCR = getattr(args, 'gcr_enable', True)
        gcr_cfg.GCR.GAL_ROUND = getattr(args, 'gcr_gal_round', 3)
        gcr_cfg.GCR.BETA1 = getattr(args, 'gcr_beta1', 0.1)
        gcr_cfg.GCR.BETA2 = getattr(args, 'gcr_beta2', 0.1)
        gcr_cfg.GCR.LAMBDA1 = getattr(args, 'gcr_lambda1', 2.0)
        gcr_cfg.GCR.LAMBDA2 = getattr(args, 'gcr_lambda2', 2.0)
        gcr_cfg.GCR.SCALE = getattr(args, 'gcr_scale', 1.0)
        gcr_cfg.GCR.MODE = getattr(args, 'gcr_mode', 'sym')
        gcr_cfg.GCR.WITH_GPU = getattr(args, 'gcr_with_gpu', True)
        
        gcr_cfg.COMMON = SimpleNamespace()
        gcr_cfg.COMMON.VERBOSE = getattr(args, 'gcr_verbose', False)
        
        rerank_cfg['gcr_cfg'] = gcr_cfg
        logger.info(f"GCR configuration: {vars(gcr_cfg.GCR)}")
        logger.info(f"GCR COMMON configuration: {vars(gcr_cfg.COMMON)}")
    
    device = "cuda"
    json_file = '/SSD_Data01/PRCV-ReID5o/data/ORBench_PRCV/val/val_queries.json'
    query_type_ranges = get_query_type_idx_range(json_file)
    test_transforms = build_transforms(img_size=args.img_size,is_train=False)
    test_gallery_dataset = GalleryDataset('/SSD_Data01/PRCV-ReID5o/data/ORBench_PRCV/val/gallery',transform=test_transforms)
    test_gallery_loader = DataLoader(test_gallery_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    model = build_model(args,num_classes=int(400*(1-args.test_size)))
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    
    if args.add_multimodal_layers:
        gfeats_dict = embedding_gfeats_with_multiembeddings(model, test_gallery_loader)
        logger.info(f"embedding_gfeats_with_multiembeddings success")
    else:
        gfeats = embedding_gfeats(model, test_gallery_loader)
        logger.info(f"embedding_gfeats success")
    
    json_file = '/SSD_Data01/PRCV-ReID5o/data/ORBench_PRCV/val/val_queries.json'
    query_type_ranges = get_query_type_idx_range(json_file)
    
    # 为每种重排序方法创建输出文件
    output_files = {}
    csv_writers = {}
    csv_files = {}
    
    for method in rerank_methods:
        output_suffix = f"_{method}" if method != 'none' else ""
        output_file = op.join(args.output_dir, f'ranking_list{output_suffix}.csv')
        output_files[method] = output_file
        
        csv_file = open(output_file, mode='w', newline='')
        csv_files[method] = csv_file
        fieldnames = ['query_idx', 'query_type', 'ranking_list_idx']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_writers[method] = writer
    
    try:
        for current_query_type, begin_idx, end_idx in query_type_ranges:
            logger.info(f"Processing {current_query_type} (idx {begin_idx}-{end_idx})")
            
            test_query_dataset = KaggleInputDataset('/SSD_Data01/PRCV-ReID5o/data/ORBench_PRCV/val/val_queries.json', begin_idx, end_idx, test_transforms)
            test_query_loader = DataLoader(test_query_dataset, batch_size=args.test_batch_size, shuffle=False)
            qfeats = embedding_qfeats(model, test_query_loader, current_query_type)
            modalities_list = current_query_type.split("_") 
            
            # 修正：保持与原始逻辑一致，直接使用 gfeats 变量
            if args.add_multimodal_layers:
                gfeats_to_combine = [gfeats_dict[m] for m in modalities_list if m in gfeats_dict and len(gfeats_dict[m]) > 0]
                if not gfeats_to_combine:
                    logger.warning(f"No features found for query type: {current_query_type}. Skipping.")
                    continue
                gfeats = sum(gfeats_to_combine) / len(gfeats_to_combine)
            
            qfeats = F.normalize(qfeats, p=2, dim=1)
            gfeats = F.normalize(gfeats, p=2, dim=1)
            
            # 应用重排序
            results = apply_reranking(qfeats, gfeats, rerank_methods, rerank_cfg, logger)
            
            if not results:
                logger.error(f"All re-ranking methods failed for {current_query_type}")
                continue
            
            # 为每种方法写入结果
            for method, indices in results.items():
                for i, query_idx in enumerate(range(begin_idx, end_idx + 1)):
                    ranking_list_idx = (indices[i, :100].cpu().numpy() + 1).tolist()
                    csv_writers[method].writerow({
                        'query_idx': query_idx+1,
                        'query_type': current_query_type,
                        'ranking_list_idx': ranking_list_idx
                    })
            
            logger.info(f"{current_query_type} completed successfully with {len(results)} methods")
    
    finally:
        # 关闭所有文件
        for csv_file in csv_files.values():
            csv_file.close()
    
    for method, output_file in output_files.items():
        logger.info(f"CSV file for {method} generated successfully: {output_file}")