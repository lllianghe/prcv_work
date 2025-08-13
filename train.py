import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator, Evaluator_OR
from utils.options import get_args
from utils.comm import get_rank, synchronize


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()
    set_seed(42)
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    print("gpu_num: ",num_gpus)
    # 检测可用gpu数量大于1则采用分布式训练方法
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    # args.output_dir = op.join(args.output_dir, args.dataset_name, f'test_fgclip')
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    # get image-text pair datasets dataloader
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes) # num_classes是训练集中行人的身份数量
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    
    model.to(device)
    

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0 #分布式的获取进程排名 如不开启分布式训练 则定义为0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master) #负责模型加载保存
    

    start_epoch = 1
    
    # 根据参数分别控制multimodal embedding和projection层的初始化
    add_embeddings = args.add_multimodal_embeddings or args.add_multimodal_layers
    add_projections = args.add_multimodal_projections or args.add_multimodal_layers
    
    # 统一的检查点加载逻辑
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume_ckpt_file}")
        # 检查是否需要严格模式
        # 严格模式条件：不需要添加任何多模态层，或者需要的层都已经存在
        has_multimodal_embeddings = hasattr(model.base_model, 'vision_model') and hasattr(model.base_model.vision_model.embeddings, 'modality_patch_embeddings') and model.base_model.vision_model.embeddings.modality_patch_embeddings
        has_multimodal_projections = hasattr(model.base_model, 'modality_visual_projections') and model.base_model.modality_visual_projections
        
        # 使用严格模式的条件：需要的层都已经存在
        # 对于每个层：要么不需要添加，要么已经存在
        need_strict_mode = (not add_embeddings or has_multimodal_embeddings) and \
                          (not add_projections or has_multimodal_projections)
        
        if need_strict_mode:
            # 严格模式：用于多模态检查点
            checkpoint = checkpointer.resume(args.resume_ckpt_file, strict=True)
            logger.info("Using strict loading mode for multimodal checkpoint")
            
            # 严格模式下恢复epoch信息
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']+1
                logger.info(f"Resuming from epoch {start_epoch}")
        else:
            # 非严格模式：用于单模态检查点，自动添加多模态层
            checkpoint = checkpointer.resume(args.resume_ckpt_file, strict=False)
            logger.info("Using non-strict loading mode for single-modal checkpoint")

            # 优化器调度器重置
            logger.info("Resetting optimizer and scheduler to use new script parameters")
            # 重新构建优化器和调度器，忽略checkpoint中的状态
            optimizer = build_optimizer(args, model)
            scheduler = build_lr_scheduler(args, optimizer)
            # 重新创建checkpointer以使用新的优化器和调度器
            checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
            # 重置训练起始epoch为1，忽略checkpoint中的epoch信息
            logger.info("Optimizer, scheduler have been reset to use new parameters")
            
            
            use_pairs = getattr(args, 'use_multimodal_layers_in_pairs', True)
            logger.info(f"Setting up multimodal layers with use_pairs={use_pairs}")
            
            if add_embeddings and not has_multimodal_embeddings:
                if hasattr(model.base_model, 'setup_multi_embeddings'):
                    model.base_model.setup_multi_embeddings(use_pairs=use_pairs)
                    logger.info(f"Setup multi embeddings with use_pairs={use_pairs}")
                else:
                    logger.warning("base_model does not support setup_multi_embeddings")
            
            if add_projections and not has_multimodal_projections:
                if hasattr(model.base_model, 'setup_multi_projections'):
                    model.base_model.setup_multi_projections(use_pairs=use_pairs)
                    logger.info(f"Setup multi projections with use_pairs={use_pairs}")
                else:
                    logger.warning("base_model does not support setup_multi_projections")
            model.to(device)
            # 多模态层已在checkpointer.load之前统一设置
            logger.info("Multimodal layers were set up before checkpoint loading")
    else:
        logger.info("Start training without loading checkpoint")
            
        if add_embeddings or add_projections:
            use_pairs = getattr(args, 'use_multimodal_layers_in_pairs', True)
            logger.info(f"Setting up multimodal layers with use_pairs={use_pairs}")
            
            if add_embeddings:
                if hasattr(model.base_model, 'setup_multi_embeddings'):
                    model.base_model.setup_multi_embeddings(use_pairs=use_pairs)
                    logger.info(f"Setup multi embeddings with use_pairs={use_pairs}")
                else:
                    logger.warning("base_model does not support setup_multi_embeddings")
            
            if add_projections:
                if hasattr(model.base_model, 'setup_multi_projections'):
                    model.base_model.setup_multi_projections(use_pairs=use_pairs)
                    logger.info(f"Setup multi projections with use_pairs={use_pairs}")
                else:
                    logger.warning("base_model does not support setup_multi_projections")
    
    # 检查是否添加了新层，如果是则重新构建优化器
    if add_embeddings or add_projections:
        logger.info("检测到新添加的多模态层，重新构建优化器...")
        
        # 重新构建优化器和调度器
        optimizer = build_optimizer(args, model)
        scheduler = build_lr_scheduler(args, optimizer)
        
        # 重新创建checkpointer
        checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
        
        logger.info(f"优化器已更新，包含 {sum(len(group['params']) for group in optimizer.param_groups)} 个参数组")
    
    model.to(device)

    if args.dataset_name == 'ORBench':
        evaluator = Evaluator_OR(val_img_loader, val_txt_loader) # 用于评估检索性能的类
    else:
        evaluator = Evaluator(val_img_loader, val_txt_loader)
    
    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)