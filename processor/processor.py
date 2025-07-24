import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator, Evaluator_OR
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import os

def plot_and_save_curves(output_dir, num_iter_per_epoch, train_loss_list, test_loss_list, r1_list, mAP_list, lr_list, eval_iters_list=None, eval_epoch_list=None,
                         r1_list_0=None, mAP_list_0=None, r1_list_1=None, mAP_list_1=None, r1_list_2=None, mAP_list_2=None,
                         r1_list_3=None, mAP_list_3=None, r1_list_4=None, mAP_list_4=None):
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 2, 1)
    plt.plot(eval_iters_list, train_loss_list, label='Train Loss')
    plt.plot(eval_iters_list, test_loss_list, label='Test Loss')
    plt.xlabel(f'*100 Eval-Steps ({num_iter_per_epoch} steps per epoch)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(num_iter_per_epoch//100))

    plt.subplot(2, 2, 2)
    if eval_epoch_list:
        plt.plot(eval_epoch_list, r1_list, label='Avg R1', linewidth=2, linestyle='--')
        if r1_list_0: plt.plot(eval_epoch_list, r1_list_0, label='R1 (4modal)')
        if r1_list_1: plt.plot(eval_epoch_list, r1_list_1, label='R1 (SK)')
        if r1_list_2: plt.plot(eval_epoch_list, r1_list_2, label='R1 (NIR)')
        if r1_list_3: plt.plot(eval_epoch_list, r1_list_3, label='R1 (CP)')
        if r1_list_4: plt.plot(eval_epoch_list, r1_list_4, label='R1 (TEXT)')
    plt.xlabel('Epoch')
    plt.ylabel('R1 Score')
    plt.legend()
    plt.title('R1 Curves')

    plt.subplot(2, 2, 3)
    if eval_epoch_list:
        plt.plot(eval_epoch_list, mAP_list, label='Avg mAP', linewidth=2, linestyle='--')
        if mAP_list_0: plt.plot(eval_epoch_list, mAP_list_0, label='mAP (4modal)')
        if mAP_list_1: plt.plot(eval_epoch_list, mAP_list_1, label='mAP (SK)')
        if mAP_list_2: plt.plot(eval_epoch_list, mAP_list_2, label='mAP (NIR)')
        if mAP_list_3: plt.plot(eval_epoch_list, mAP_list_3, label='mAP (CP)')
        if mAP_list_4: plt.plot(eval_epoch_list, mAP_list_4, label='mAP (TEXT)')
    plt.xlabel('Epoch')
    plt.ylabel('mAP Score')
    plt.legend()
    plt.title('mAP Curves')

    plt.subplot(2, 2, 4)
    plt.plot(eval_iters_list, lr_list, label='Learning Rate')
    plt.xlabel(f'*100 Eval-Steps ({num_iter_per_epoch} steps per epoch)')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title('Learning Rate Curve')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'curves.png')
    plt.savefig(save_path)
    plt.close()


def do_train(start_epoch, args, model, train_loader, test_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    num_iter_per_epoch = len(train_loader)
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    # meters是计量器可自动更新 平均值
    meters = {
        "loss": AverageMeter(),
        "test_loss": AverageMeter(), # add for val loss
        "multi_modal_contrastive_sdm_loss": AverageMeter(),
        "sk_sdm_loss": AverageMeter(),
        "nir_sdm_loss": AverageMeter(),
        "cp_sdm_loss": AverageMeter(),
        "text_sdm_loss": AverageMeter(),
        "multi_modal_contrastive_itc_loss": AverageMeter(),
        "sk_itc_loss": AverageMeter(),
        "nir_itc_loss": AverageMeter(),
        "cp_itc_loss": AverageMeter(),
        "text_itc_loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_r1 = 0.0
    best_loss = float('inf')
    best_mAP = 0.0

    # Lists to store metrics for plotting
    eval_iters_list = [] # for plotting loss curves
    eval_count = 0
    train_loss_list = []
    test_loss_list = []
    lr_list = []

    r1_list = []
    mAP_list = []
    eval_epoch_list = [] # for plotting r1 and mAP curves

    r1_list_0, mAP_list_0 = [], []
    r1_list_1, mAP_list_1 = [], []
    r1_list_2, mAP_list_2 = [], []
    r1_list_3, mAP_list_3 = [], []
    r1_list_4, mAP_list_4 = [], []

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 打印当前显存使用情况
            allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # 转换为 GB
            cached_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)  # 转换为 GB
            # logger.info(f"Iteration {n_iter + 1}/{len(train_loader)} - Allocated memory: {allocated_memory:.2f} GB, Cached memory: {cached_memory:.2f} GB")
            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k]) # 计算损失函数 损失在模型中计算好了
            if args.dataset_name == 'ORBench':
                batch_size = batch['vis_images'].shape[0]
            else: batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['multi_modal_contrastive_sdm_loss'].update(ret.get('multi_modal_contrastive_sdm_loss', 0), batch_size)
            meters['sk_sdm_loss'].update(ret.get('sk_sdm_Loss', 0), batch_size)
            meters['nir_sdm_loss'].update(ret.get('nir_sdm_Loss', 0), batch_size)
            meters['cp_sdm_loss'].update(ret.get('cp_sdm_Loss', 0), batch_size)
            meters['text_sdm_loss'].update(ret.get('text_sdm_Loss', 0), batch_size)
            meters['multi_modal_contrastive_itc_loss'].update(ret.get('multi_modal_contrastive_itc_loss', 0), batch_size)
            meters['sk_itc_loss'].update(ret.get('sk_itc_Loss', 0), batch_size)
            meters['nir_itc_loss'].update(ret.get('nir_itc_Loss', 0), batch_size)
            meters['cp_itc_loss'].update(ret.get('cp_itc_Loss', 0), batch_size)
            meters['text_itc_loss'].update(ret.get('text_itc_Loss', 0), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
            synchronize() # 分布式计算相关

            # 计算测试集损失
            if (n_iter + 1) % log_period == 0:
                model.eval()
                with torch.no_grad():
                    test_loader_iter = iter(test_loader)
                    for batch_test in test_loader_iter:
                        batch_test = {k: v.to(device) for k, v in batch_test.items()}
                        ret_test = model(batch_test)
                        total_loss_test = sum([v for k, v in ret_test.items() if "loss" in k])
                        batch_size_test = batch_test['vis_images'].shape[0]
                        meters['test_loss'].update(total_loss_test.item(), batch_size_test)
                        break # 只要一个batch
                model.train()
                
                train_loss_list.append(meters['loss'].avg)
                test_loss_list.append(meters['test_loss'].avg)
                lr_list.append(scheduler.get_lr()[0])
                eval_count += 1
                eval_iters_list.append(eval_count)
                plot_and_save_curves(args.output_dir, num_iter_per_epoch, train_loss_list, test_loss_list, r1_list, mAP_list, lr_list, eval_iters_list, eval_epoch_list=eval_epoch_list,
                                     r1_list_0=r1_list_0, mAP_list_0=mAP_list_0, r1_list_1=r1_list_1, mAP_list_1=mAP_list_1, r1_list_2=r1_list_2, mAP_list_2=mAP_list_2, r1_list_3=r1_list_3, mAP_list_3=mAP_list_3, r1_list_4=r1_list_4, mAP_list_4=mAP_list_4)

                if meters['test_loss'].avg < best_loss:
                    best_loss = meters['test_loss'].avg
                    arguments["best_loss_epoch"] = epoch
                    checkpointer.save("best_loss", **arguments)


            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

            if (n_iter + 1) % args.schedule_steps == 0:
                scheduler.step(args.schedule_steps)

        
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        # scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0 and epoch >= args.val_start_epoch:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    r1 = evaluator.eval(model.module.eval())
                else:
                    r10,mAP0 = evaluator.eval(model.eval(), modalities="fourmodal_SK_NIR_CP_TEXT")
                    r11,mAP1 = evaluator.eval(model.eval(), modalities="onemodal_SK")
                    r12,mAP2 = evaluator.eval(model.eval(), modalities="onemodal_NIR")
                    r13,mAP3 = evaluator.eval(model.eval(), modalities="onemodal_CP")
                    r14,mAP4 =  evaluator.eval(model.eval(), modalities="onemodal_TEXT")
                    r1 = (r11+r12+r13+r14)/4
                    mAP = (mAP1+mAP2+mAP3+mAP4)/4
                    print(f'R1: {r1:.4f}\nmAP: {mAP:.4f}')


                eval_epoch_list.append(epoch)
                r1_list.append(r1)
                mAP_list.append(mAP)

                r1_list_0.append(r10)
                mAP_list_0.append(mAP0)
                r1_list_1.append(r11)
                mAP_list_1.append(mAP1)
                r1_list_2.append(r12)
                mAP_list_2.append(mAP2)
                r1_list_3.append(r13)
                mAP_list_3.append(mAP3)
                r1_list_4.append(r14)
                mAP_list_4.append(mAP4)

                plot_and_save_curves(args.output_dir, num_iter_per_epoch, train_loss_list, test_loss_list, r1_list, mAP_list, lr_list, eval_iters_list, eval_epoch_list=eval_epoch_list,
                                     r1_list_0=r1_list_0, mAP_list_0=mAP_list_0, r1_list_1=r1_list_1, mAP_list_1=mAP_list_1, r1_list_2=r1_list_2, mAP_list_2=mAP_list_2, r1_list_3=r1_list_3, mAP_list_3=mAP_list_3, r1_list_4=r1_list_4, mAP_list_4=mAP_list_4)

                torch.cuda.empty_cache()
                if best_r1 < r10:
                    best_r1 = r10
                    arguments["best_r1_epoch"] = epoch
                    checkpointer.save("best_r1", **arguments)
                if best_mAP < mAP:
                    best_mAP = mAP
                    arguments["best_mAP_epoch"] = epoch
                    checkpointer.save("best_mAP", **arguments)
                    
                logger.info(f"best loss: {best_loss} at epoch {arguments['best_loss_epoch']}")
                logger.info(f"best R1: {best_r1} at epoch {arguments['best_r1_epoch']}")
                logger.info(f"best mAP: {best_mAP} at epoch {arguments['best_mAP_epoch']}")


def do_inference(model, test_img_loader, test_txt_loader,modalities):

    logger = logging.getLogger(f"IRRA.test:")
    logger.info("Enter inferencing")
    # evaluator = Evaluator(test_img_loader, test_txt_loader)
    evaluator = Evaluator_OR(test_img_loader, test_txt_loader)
    r1,map = evaluator.eval(model.eval(),modalities=modalities)
    return r1,map
