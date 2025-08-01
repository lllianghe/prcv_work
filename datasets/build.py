import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset
from .bases_or import ORBenchTrainDataset,ORBenchQueryDataset,ORBenchGalleryDataset
from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .orbench import ORBENCH

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid, 'ORBench': ORBENCH}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir,test_size = args.test_size)
    num_classes = len(dataset.train_id_container) # 训练时用的pid数量
    
    if args.training:
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)
        if args.dataset_name== 'ORBench':
            train_set = ORBenchTrainDataset(dataset.train,
                            train_transforms,
                            text_length=args.text_length)
        else:
            if args.MLM:
                train_set = ImageTextMLMDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
            else:
                train_set = ImageTextDataset(dataset.train,
                                    train_transforms,
                                    text_length=args.text_length)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                data_sampler = RandomIdentitySampler_DDP(dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
                train_loader = DataLoader(train_set,
                                          batch_sampler=batch_sampler,
                                          num_workers=num_workers,
                                          collate_fn=collate)
            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate,
                                          drop_last=args.drop_last)
        elif args.sampler == 'random': # 使用ramdom sampler
            # TODO add distributed condition
            logger.info('using random sampler')
            # 每次训练时都会将数据随机打乱
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate,
                                          drop_last=args.drop_last)
            # collate将多个样本整理成一个批次
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set 将测试集当验证集 这部分是要改的
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        if args.dataset_name == 'ORBench':
            val_img_set = ORBenchGalleryDataset(ds['image_pids'],ds['vis_img_paths'],val_transforms)
            val_txt_set = ORBenchQueryDataset(ds['caption_pids'],
                                     ds['captions'], 
                                     ds['cp_paths'], 
                                     ds['sk_paths'], 
                                     ds['nir_paths'],
                                     transform = val_transforms,
                                     text_length=args.text_length)
        else:
            val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    val_transforms)
            val_txt_set = TextDataset(ds['caption_pids'],
                                    ds['captions'],
                                    text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        # print(len(train_loader), len(val_img_loader), len(val_txt_loader))
        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing 
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test #还是用的是测试集
        if args.dataset_name == 'ORBench':
            test_img_set = ORBenchGalleryDataset(ds['image_pids'],ds['vis_img_paths'],test_transforms)
            test_txt_set = ORBenchQueryDataset(ds['caption_pids'],
                                     ds['captions'], 
                                     ds['cp_paths'], 
                                     ds['sk_paths'], 
                                     ds['nir_paths'],
                                     transform = test_transforms,
                                     text_length=args.text_length)
        else:
            test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                        test_transforms)
            test_txt_set = TextDataset(ds['caption_pids'],
                                    ds['captions'],
                                    text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        return test_img_loader, test_txt_loader, num_classes
