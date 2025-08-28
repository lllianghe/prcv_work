from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
from .re_ranking import fast_gcrv_image
from .cmc import ReRank


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def rank_with_distance(dist_matrix, q_pids, g_pids, max_rank=10, get_mAP=True):
    """使用距离矩阵进行排序和评估"""
    indices = np.argsort(dist_matrix, axis=1)  # 距离矩阵，升序排列
    pred_labels = g_pids[indices][:, :max_rank]  # q * k
    matches = (pred_labels == q_pids.reshape(-1, 1))  # q * k

    all_cmc = np.cumsum(matches, axis=1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.astype(np.float32).mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = np.cumsum(matches, axis=1)  # q * k

    inp = []
    for i, match_row in enumerate(matches):
        match_indices = np.where(match_row)[0]
        if len(match_indices) > 0:
            inp.append(tmp_cmc[i][match_indices[-1]] / (match_indices[-1] + 1.))
        else:
            inp.append(0.)
    mINP = np.mean(inp) * 100

    tmp_cmc_norm = np.zeros_like(tmp_cmc, dtype=np.float32)
    for i in range(tmp_cmc.shape[1]):
        tmp_cmc_norm[:, i] = tmp_cmc[:, i] / (i + 1.0)
    tmp_cmc_norm = tmp_cmc_norm * matches
    AP = tmp_cmc_norm.sum(1) / np.maximum(num_rel, 1e-8)  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        table.hrules = 1
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0]



class Evaluator_OR():
    def __init__(self, img_loader, txt_loader, use_multimodal_layers_in_pairs=False):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")
        self.use_multimodal_layers_in_pairs = use_multimodal_layers_in_pairs

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids = [], []
        # A dictionary to store features for each modality
        # query: four modalities always
        qfeats_dict = {'TEXT': [], 'CP': [], 'SK': [], 'NIR': []}
        # gallery: either four branches (pairs=True) or single 'VIS' branch (pairs=False)
        gfeats_dict = {'TEXT': [], 'CP': [], 'SK': [], 'NIR': [], 'VIS': []}

        # text/query loader
        for pid, cp_img, sk_img, nir_img, caption in self.txt_loader:
            qids.append(pid.view(-1))
            with torch.no_grad():
                # TEXT
                caption = caption.to(device)
                text_feat = model.encode_text(caption)
                qfeats_dict['TEXT'].append(text_feat)
                # CP
                cp_img = cp_img.to(device)
                cp_feat = model.encode_image(cp_img, modality='cp')
                qfeats_dict['CP'].append(cp_feat)
                # SK
                sk_img = sk_img.to(device)
                sk_feat = model.encode_image(sk_img, modality='sk')
                qfeats_dict['SK'].append(sk_feat)
                # NIR
                nir_img = nir_img.to(device)
                nir_feat = model.encode_image(nir_img, modality='nir')
                qfeats_dict['NIR'].append(nir_feat)

        qids = torch.cat(qids, 0)
        for modality in qfeats_dict:
            if qfeats_dict[modality] and len(qfeats_dict[modality]) > 0:
                 qfeats_dict[modality] = torch.cat(qfeats_dict[modality], 0)

        # image/gallery loader
        for pid, img in self.img_loader:
            gids.append(pid.view(-1))
            with torch.no_grad():
                img = img.to(device)
                if self.use_multimodal_layers_in_pairs:
                    # produce four gallery branches to match corresponding queries
                    text_feat = model.encode_image(img, modality='vis')
                    gfeats_dict['TEXT'].append(text_feat)
                    cp_feat = model.encode_image(img, modality='cp')
                    gfeats_dict['CP'].append(cp_feat)
                    sk_feat = model.encode_image(img, modality='sk')
                    gfeats_dict['SK'].append(sk_feat)
                    nir_feat = model.encode_image(img, modality='nir')
                    gfeats_dict['NIR'].append(nir_feat)
                else:
                    # single VIS gallery - 通过模态回退机制，所有模态调用都会使用同一个默认 visual_projection
                    vis_feat = model.encode_image(img, modality='vis')  # 回退到默认层
                    gfeats_dict['VIS'].append(vis_feat)
        gids = torch.cat(gids, 0)
        for modality in gfeats_dict:
            if gfeats_dict[modality] and len(gfeats_dict[modality]) > 0:
                 gfeats_dict[modality] = torch.cat(gfeats_dict[modality], 0)

        return qfeats_dict, gfeats_dict, qids, gids
    
    def eval(self, model, i2t_metric=False, rerank_method=None, rerank_cfg=None, modalities=["onemodal_SK", "onemodal_NIR", "onemodal_CP", "onemodal_TEXT", '' ,"twomodal_SK_NIR", "twomodal_SK_CP","twomodal_SK_TEXT", "twomodal_NIR_CP", "twomodal_NIR_TEXT", "twomodal_CP_TEXT", '', "threemodal_SK_NIR_CP", "threemodal_SK_NIR_TEXT", "threemodal_SK_CP_TEXT", "threemodal_NIR_CP_TEXT", '', "fourmodal_SK_TEXT_CP_NIR"]):
        """
        Args:
            rerank_method: None, 'rerank', 'gcr' - 选择重排序方法
            rerank_cfg: 重排序配置参数
        """
        
        if len(self.img_loader)==0:
            print("No data! Skip Evaluating.\n")
            return 0.0,0.0
        
        # Step 1: Compute all embeddings once
        qfeats_dict, gfeats_dict, qids, gids = self._compute_embedding(model)

        all_r1s = []
        all_mAPs = []
        result_rows = []

        # Step 2: Loop through each evaluation strategy
        for modality_strategy in modalities:
            if modality_strategy == '':
                result_rows.append([''] * 6)
                continue

            modalities_list = modality_strategy.split("_")[1:] # e.g., from "fourmodal_SK_TEXT_CP_NIR" to ['SK', 'TEXT', 'CP', 'NIR']
            
            # Combine features for the current strategy (query)
            feats_to_combine = [qfeats_dict[m] for m in modalities_list if m in qfeats_dict and len(qfeats_dict[m]) > 0]
            if not feats_to_combine:
                self.logger.warning(f"No features found for modality strategy: {modality_strategy}. Skipping.")
                continue
            qfeats = sum(feats_to_combine) / len(feats_to_combine)
            
            # Combine gallery features based on pairing flag
            if self.use_multimodal_layers_in_pairs:
                gfeats_to_combine = [gfeats_dict[m] for m in modalities_list if m in gfeats_dict and len(gfeats_dict[m]) > 0]
                if not gfeats_to_combine:
                    self.logger.warning(f"No gallery features found for modality strategy: {modality_strategy}. Skipping.")
                    continue
                gfeats = sum(gfeats_to_combine) / len(gfeats_to_combine)
            else:
                gfeats = gfeats_dict["VIS"]

            qfeats = F.normalize(qfeats, p=2, dim=1)  # query features
            gfeats = F.normalize(gfeats, p=2, dim=1)  # gallery features

            # Apply re-ranking if specified
            if rerank_method is None:
                # Original similarity-based ranking
                similarity = qfeats @ gfeats.t()
                t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
                t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
            elif rerank_method == 'rerank':
                # K-reciprocal re-ranking
                k1 = rerank_cfg.get('k1', 20) if rerank_cfg else 20
                k2 = rerank_cfg.get('k2', 6) if rerank_cfg else 6
                lambda_value = rerank_cfg.get('lambda_value', 0.3) if rerank_cfg else 0.3
                
                self.logger.info(f"Applying k-reciprocal re-ranking for {modality_strategy} with k1={k1}, k2={k2}, lambda={lambda_value}")
                dist_matrix = ReRank(qfeats.cpu().numpy(), gfeats.cpu().numpy(), k1=k1, k2=k2, lambda_value=lambda_value)
                t2i_cmc, t2i_mAP, t2i_mINP, _ = rank_with_distance(dist_matrix, qids.cpu().numpy(), gids.cpu().numpy(), max_rank=10, get_mAP=True)
            elif rerank_method == 'gcr':
                # GCR re-ranking
                if rerank_cfg is None:
                    self.logger.error("GCR configuration is required for GCR re-ranking")
                    continue
                
                self.logger.info(f"Applying GCR re-ranking for {modality_strategy}")
                # 准备GCR需要的数据格式
                qids_np = qids.cpu().numpy()
                gids_np = gids.cpu().numpy()
                qfeats_np = qfeats.cpu()
                gfeats_np = gfeats.cpu()
                
                # 创建伪tracks（如果不需要可以设为0）
                qids_tracks = np.zeros_like(qids_np)
                gids_tracks = np.zeros_like(gids_np)
                
                all_data = [qfeats_np, qids_np, qids_tracks, gfeats_np, gids_np, gids_tracks]
                dist_matrix, _, _ = fast_gcrv_image(rerank_cfg, all_data)
                t2i_cmc, t2i_mAP, t2i_mINP, _ = rank_with_distance(dist_matrix, qids_np, gids_np, max_rank=10, get_mAP=True)
            
            all_r1s.append(t2i_cmc[0])
            all_mAPs.append(t2i_mAP)

            # Add to table
            task_name = f"{modality_strategy}" + (f"_{rerank_method}" if rerank_method else "")
            result_rows.append([task_name, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

            if i2t_metric:
                if rerank_method is None:
                    similarity = qfeats @ gfeats.t()
                    i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
                    i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                elif rerank_method == 'rerank':
                    # For i2t, swap query and gallery
                    dist_matrix_i2t = ReRank(gfeats.cpu().numpy(), qfeats.cpu().numpy(), k1=k1, k2=k2, lambda_value=lambda_value)
                    i2t_cmc, i2t_mAP, i2t_mINP, _ = rank_with_distance(dist_matrix_i2t, gids.cpu().numpy(), qids.cpu().numpy(), max_rank=10, get_mAP=True)
                elif rerank_method == 'gcr':
                    # For i2t, swap query and gallery
                    all_data_i2t = [gfeats_np, gids_np, gids_tracks, qfeats_np, qids_np, qids_tracks]
                    dist_matrix_i2t, _, _ = fast_gcrv_image(rerank_cfg, all_data_i2t)
                    i2t_cmc, i2t_mAP, i2t_mINP, _ = rank_with_distance(dist_matrix_i2t, gids_np, qids_np, max_rank=10, get_mAP=True)
                
                i2t_task_name = f'i2t_{modality_strategy}' + (f"_{rerank_method}" if rerank_method else "")
                result_rows.append([i2t_task_name, i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        avg_r1 = np.mean(all_r1s)
        avg_mAP = np.mean(all_mAPs)
        avg_task_name = f'Average' + (f"_{rerank_method}" if rerank_method else "")
        table.add_row([avg_task_name, avg_r1, '-', '-', avg_mAP, '-'])
        table.add_row([''] * 6)

        for row in result_rows:
            table.add_row(row)

        # Formatting and logging
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
        table.hrules = 1
        self.logger.info('\n' + str(table))
        
        # 提取四种单模态的mAP
        single_modal_mAPs = {}
        for row in result_rows:
            if len(row) > 0 and row[0].startswith('onemodal_'):
                modal_name = row[0].split('_')[1]  # 提取模态名称
                single_modal_mAPs[modal_name] = row[4]  # mAP值在第5列（索引4）
        
        return avg_r1, avg_mAP, single_modal_mAPs