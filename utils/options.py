import argparse



def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="irra", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", type=int, default=30)
    parser.add_argument("--eval_period", type=float, default=1)
    parser.add_argument("--val_dataset",default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=True, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=False, action='store_true', help="whether to use Mask Language Modeling dataset") #暂时删除mlm功能 下面的loss_names也删了mlm

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='sdm+id', help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=400)
    parser.add_argument("--annealing_epochs", type=int, default=2000) # 新的参数, epoch的含义是step
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--scheduler_period", type=int, default=50)
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default='ORBench', help="[CUHK-PEDES, ICFG-PEDES, RSTPReid, ORBench]")
    # parser.add_argument("--dataset_name", default='CUHK-PEDES', help="[CUHK-PEDES, ICFG-PEDES, RSTPReid, ORBench]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [identity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="data_files")
    parser.add_argument("--batch_size", type=int, default=32) # batch_size原来是64
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false') # 意思是如果有--test args.trainging就会被设置为true
    parser.add_argument("--test_size", type=float,default=150/400)  # 测试集占总样本的比例
    parser.add_argument("--drop_last", type=bool, default=False)

    args = parser.parse_args()

    return args