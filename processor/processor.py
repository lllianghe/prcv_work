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

def plot_and_save_curves(output_dir, num_iter_per_epoch, train_loss_list, mAP_list, lr_list, log_period, eval_iters_list=None, eval_epoch_list=None):
    plt.figure(figsize=(18, 5))

    # Plotting Training Loss
    plt.subplot(1, 3, 1)
    plt.plot(eval_iters_list, train_loss_list, label='Train Loss')
    plt.xlabel(f'x{log_period} Steps ({num_iter_per_epoch} per epoch)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.grid(True)

    # Plotting mAP Score
    plt.subplot(1, 3, 2)
    if eval_epoch_list:
        plt.plot(eval_epoch_list, mAP_list, label='mAP', marker='o')
    plt.xlabel('Eval-Epoch')
    plt.ylabel('mAP Score')
    plt.legend()
    plt.title('mAP Curve')
    plt.grid(True)

    # Plotting Learning Rate
    plt.subplot(1, 3, 3)
    plt.plot(eval_iters_list, lr_list, label='Learning Rate')
    plt.xlabel(f'x{log_period} Steps ({num_iter_per_epoch} per epoch)')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title('Learning Rate Curve')
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'curves.png')
    plt.savefig(save_path)
    plt.close()


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, train_data_sampler=None, device=None):

    # log_period = args.log_period
    log_period = len(train_loader)
    scheduler_period = len(train_loader)

    eval_period = args.eval_period
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    # meters是计量器可自动更新 平均值
    meters = {
        "loss": AverageMeter(),
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

    best_mAP = 0.0

    # Lists to store metrics for plotting
    train_loss_list = []
    mAP_list = []
    lr_list = []
    eval_iters_list = []
    eval_epoch_list = []
    eval_count = 0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        train_data_sampler.set_epoch(epoch) if train_data_sampler else None
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
            
            optimizer.zero_grad()

            ret = model(batch)

            total_loss = sum([v for k, v in ret.items() if "loss" in k]) # 计算损失函数 multi_modal_contrastive_loss损失在模型中计算好了, 并且已经成功detach

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

            if total_loss.requires_grad:
                total_loss.backward()
            optimizer.step()

            
            if (n_iter + 1) % log_period == 0 and get_rank() == 0:
                eval_count += 1
                train_loss_list.append(meters['loss'].avg)
                lr_list.append(scheduler.get_lr()[0])
                eval_iters_list.append(eval_count)
                plot_and_save_curves(args.output_dir, len(train_loader), train_loss_list, mAP_list, lr_list, log_period, eval_iters_list, eval_epoch_list)

                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
            
            if (n_iter + 1) % scheduler_period == 0:
                scheduler.step(args.scheduler_period)
                # print(f"Epoch {epoch}, Iteration {n_iter + 1}, Lr: {scheduler.get_lr()[0]:.2e}")
            
        if get_rank() == 0:
            tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
            tb_writer.add_scalar('temperature', ret['temperature'], epoch)
            for k, v in meters.items():
                if v.avg > 0:
                    tb_writer.add_scalar(k, v.avg, epoch)

        # print speed
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))

        # evalue
        if epoch % eval_period == 0 and args.test_size > 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))

                if args.distributed:
                    r1, mAP = evaluator.eval(model.module.eval())
                else:
                    r1, mAP = evaluator.eval(model.eval())
                
                eval_epoch_list.append(epoch)
                mAP_list.append(mAP)
                plot_and_save_curves(args.output_dir, len(train_loader), train_loss_list, mAP_list, lr_list, log_period, eval_iters_list, eval_epoch_list)

                torch.cuda.empty_cache()
                if best_mAP < mAP:
                    best_mAP = mAP
                    arguments["best_mAP_epoch"] = epoch
                    checkpointer.save("best", **arguments)
                logger.info(f"best mAP: {best_mAP} at epoch {arguments['best_mAP_epoch']}")

            if args.distributed:
                synchronize()


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger(f"IRRA.test:")
    logger.info("Enter inferencing")
    evaluator = Evaluator_OR(test_img_loader, test_txt_loader)

    return evaluator.eval(model.eval())
