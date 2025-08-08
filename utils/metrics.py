from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


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
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids = [], []
        # A dictionary to store features for each modality
        gfeats_dict = {'TEXT': [], 'CP': [], 'SK': [], 'NIR': []}
        qfeats_dict = {'TEXT': [], 'CP': [], 'SK': [], 'NIR': []}

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
        gids = torch.cat(gids, 0)
        for modality in gfeats_dict:
            if gfeats_dict[modality] and len(gfeats_dict[modality]) > 0:
                 gfeats_dict[modality] = torch.cat(gfeats_dict[modality], 0)

        return qfeats_dict, gfeats_dict, qids, gids
    
    def eval(self, model, i2t_metric=False, modalities=["onemodal_SK", "onemodal_NIR", "onemodal_CP", "onemodal_TEXT", '' ,"twomodal_SK_NIR", "twomodal_SK_CP","twomodal_SK_TEXT", "twomodal_NIR_CP", "twomodal_NIR_TEXT", "twomodal_CP_TEXT", '', "threemodal_SK_NIR_CP", "threemodal_SK_NIR_TEXT", "threemodal_SK_CP_TEXT", "threemodal_NIR_CP_TEXT", '', "fourmodal_SK_TEXT_CP_NIR"]):
        
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
            
            # Combine features for the current strategy
            feats_to_combine = [qfeats_dict[m] for m in modalities_list if m in qfeats_dict and len(qfeats_dict[m]) > 0]
            if not feats_to_combine:
                self.logger.warning(f"No features found for modality strategy: {modality_strategy}. Skipping.")
                continue
            qfeats = sum(feats_to_combine) / len(feats_to_combine)
            
            gfeats_to_combine = [gfeats_dict[m] for m in modalities_list if m in gfeats_dict and len(gfeats_dict[m]) > 0]
            if not gfeats_to_combine:
                self.logger.warning(f"No features found for modality strategy: {modality_strategy}. Skipping.")
                continue
            gfeats = sum(gfeats_to_combine) / len(gfeats_to_combine)

            qfeats = F.normalize(qfeats, p=2, dim=1)  # text features (query)
            gfeats = F.normalize(gfeats, p=2, dim=1)  # image features (gallery)

            # Calculate similarity
            similarity = qfeats @ gfeats.t()

            # Rank and get metrics
            t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
            t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
            
            all_r1s.append(t2i_cmc[0])
            all_mAPs.append(t2i_mAP)

            # Add to table
            result_rows.append([modality_strategy, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])


            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                result_rows.append([f'i2t_{modality_strategy}', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        avg_r1 = np.mean(all_r1s)
        avg_mAP = np.mean(all_mAPs)
        table.add_row(['Average', avg_r1, '-', '-', avg_mAP, '-'])
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
        
        return avg_r1, avg_mAP
