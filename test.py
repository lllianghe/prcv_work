import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from prettytable import PrettyTable
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs

import random
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    set_seed(1)

    parser = argparse.ArgumentParser(description="irra Test")
    parser.add_argument("--config_file", default=
                                                '/SSD_Data01/myf/research/PRCV/fgclip_model/prcv_work_branch/logs/ORBench/20250723_045537_irra/configs.yaml'
                                                )
    # parser.add_argument("--config_file", default='logs/ORBench/20250715_021439_irra/configs.yaml') #这是fgclip的模型
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.test_batch_size = 256
    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    names = {'best_mAP.pth', 'best_loss.pth', 'best_r1.pth', 'best_top1.pth', 'best.pth'}
    for n in names:
        path = op.join(args.output_dir, f'{n}')
        if not op.exists(path):
            continue
        print('\n',f'Evalue for {n} :')
        checkpointer.load(f=path)
        model.to(device)
        modalities = "onemodal_SK"
        do_inference(model, test_img_loader, test_txt_loader,"fourmodal_SK_NIR_CP_TEXT")
        do_inference(model, test_img_loader, test_txt_loader,"onemodal_SK")
        do_inference(model, test_img_loader, test_txt_loader,"onemodal_NIR")
        do_inference(model, test_img_loader, test_txt_loader,"onemodal_CP")
        do_inference(model, test_img_loader, test_txt_loader,"onemodal_TEXT")