import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')
    
    # 检查是否启用了多模态projection层
    add_projections = getattr(args, 'add_multimodal_projections', False) or getattr(args, 'add_multimodal_layers', False)
    if add_projections:
        print(f'Using 4x learning rate for projection layers when add_multimodal_projections is enabled')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay

        if "cross" in key:
            # use large learning rate for random initialized cross modal module
            lr =  args.lr * args.lr_factor # default 5.0
        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        if "classifier" in key or "mlm_head" in key:
            lr = args.lr * args.lr_factor
        
        # 当启用add_multimodal_projections或add_multimodal_layers时，为特定的projection层设置4倍学习率
        if add_projections:
            # 只针对fgclip.py中定义的特定projection层：text_projection和modality_visual_projections
            if ("text_projection" in key or "modality_visual_projections" in key):
                lr = args.lr * 4.0  # 将特定projection层学习率提高4倍
                print(f"Setting FGCLIPModel projection layer '{key}' learning rate to {lr} (4x base rate)")
        
        # weight_decay用于l2正则化来防止过拟合
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_method=args.warmup_method,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        annealing_epochs=args.annealing_epochs, # 新的参数, epoch=step
        min_lr=args.min_lr,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
        step_size=args.step_size
        
    )
