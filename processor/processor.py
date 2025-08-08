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
import wandb

from torch.amp import GradScaler


def plot_and_save_curves(output_dir, num_iter_per_epoch, train_loss_list, mAP_list, lr_list, log_period, eval_iters_list=None, eval_epoch_list=None, loss_dict=None, single_modal_mAPs_history=None):
    # 绘制损失曲线图
    plt.figure(figsize=(20, 10))
    
    # 第一行：损失曲线
    plt.subplot(2, 2, 1)
    if len(train_loss_list) > 0 and len(eval_iters_list) > 0:
        # Convert CUDA tensors to CPU if needed
        train_loss_values = train_loss_list
        if len(train_loss_values) > 0 and hasattr(train_loss_values[0], 'cpu'):
            train_loss_values = [val.cpu().item() if hasattr(val, 'cpu') else val for val in train_loss_values]
        plt.plot(eval_iters_list, train_loss_values, label='Total Loss', linewidth=2)
    if loss_dict:
        for loss_name, loss_values in loss_dict.items():
            if len(loss_values) > 0 and len(eval_iters_list) >= len(loss_values) and max(loss_values) > 0:
                x_values = eval_iters_list[-len(loss_values):] if len(loss_values) <= len(eval_iters_list) else eval_iters_list
                y_values = loss_values[-len(x_values):] if len(loss_values) > len(x_values) else loss_values
                # Convert CUDA tensors to CPU if needed
                if hasattr(y_values[0], 'cpu'):
                    y_values = [val.cpu().item() if hasattr(val, 'cpu') else val for val in y_values]
                plt.plot(x_values, y_values, label=loss_name, alpha=0.7)
    plt.xlabel(f'x{log_period} Steps ({num_iter_per_epoch} per epoch)')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('All Training Loss Curves')
    plt.grid(True)
    
    # 第一行：学习率曲线
    plt.subplot(2, 2, 2)
    if len(lr_list) > 0 and len(eval_iters_list) > 0:
        # Convert CUDA tensors to CPU if needed
        lr_values = lr_list
        if len(lr_values) > 0 and hasattr(lr_values[0], 'cpu'):
            lr_values = [val.cpu().item() if hasattr(val, 'cpu') else val for val in lr_values]
        plt.plot(eval_iters_list, lr_values, label='Learning Rate', color='red')
    plt.xlabel(f'x{log_period} Steps ({num_iter_per_epoch} per epoch)')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title('Learning Rate Curve')
    plt.grid(True)
    
    # 第二行：mAP曲线
    plt.subplot(2, 2, 3)
    if eval_epoch_list and len(mAP_list) > 0:
        # Convert CUDA tensors to CPU if needed
        mAP_values = mAP_list
        if len(mAP_values) > 0 and hasattr(mAP_values[0], 'cpu'):
            mAP_values = [val.cpu().item() if hasattr(val, 'cpu') else val for val in mAP_values]
        plt.plot(eval_epoch_list, mAP_values, label='Average mAP', marker='o', linewidth=2, markersize=6)
    
    # 绘制单模态mAP曲线
    if single_modal_mAPs_history:
        colors = ['blue', 'green', 'red', 'orange']
        markers = ['s', '^', 'v', 'd']
        modal_names = ['SK', 'NIR', 'CP', 'TEXT']
        
        for i, modal in enumerate(modal_names):
            if modal in single_modal_mAPs_history and len(single_modal_mAPs_history[modal]) > 0:
                # Convert CUDA tensors to CPU if needed
                modal_values = single_modal_mAPs_history[modal]
                if len(modal_values) > 0 and hasattr(modal_values[0], 'cpu'):
                    modal_values = [val.cpu().item() if hasattr(val, 'cpu') else val for val in modal_values]
                plt.plot(eval_epoch_list[-len(modal_values):], 
                        modal_values, 
                        label=f'{modal} mAP', 
                        marker=markers[i], 
                        color=colors[i], 
                        alpha=0.8,
                        markersize=4)
        
        # 计算并绘制四种单模态的平均mAP
        if all(modal in single_modal_mAPs_history for modal in modal_names):
            min_len = min(len(single_modal_mAPs_history[modal]) for modal in modal_names)
            if min_len > 0:
                avg_single_modal_mAPs = []
                for i in range(min_len):
                    modal_values = []
                    for modal in modal_names:
                        val = single_modal_mAPs_history[modal][-min_len+i]
                        # Convert CUDA tensor to CPU if needed
                        if hasattr(val, 'cpu'):
                            val = val.cpu().item()
                        modal_values.append(val)
                    avg_mAP = sum(modal_values) / 4
                    avg_single_modal_mAPs.append(avg_mAP)
                plt.plot(eval_epoch_list[-min_len:], avg_single_modal_mAPs, 
                        label='Avg Single Modal mAP', 
                        marker='*', 
                        color='purple', 
                        linewidth=2,
                        markersize=8)
    
    plt.xlabel('Eval-Epoch')
    plt.ylabel('mAP Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('mAP Curves')
    plt.grid(True)
    
    # 第二行：其他指标曲线
    plt.subplot(2, 2, 4)
    # 可以绘制准确率等其他指标
    plt.text(0.5, 0.5, 'Reserved for\nOther Metrics\n(Accuracy, etc.)', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=12)
    plt.title('Other Metrics')
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    # log_period = args.log_period
    log_period = len(train_loader)
    scheduler_period = len(train_loader)

    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    wandb.log({"message": "start training"})

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
    
    # 记录各种损失的历史
    loss_dict = {
        'multi_modal_contrastive_sdm_loss': [],
        'sk_sdm_loss': [],
        'nir_sdm_loss': [],
        'cp_sdm_loss': [],
        'text_sdm_loss': [],
        'multi_modal_contrastive_itc_loss': [],
        'sk_itc_loss': [],
        'nir_itc_loss': [],
        'cp_itc_loss': [],
        'text_itc_loss': [],
        'sdm_loss': [],
        'itc_loss': [],
        'id_loss': [],
        'mlm_loss': []
    }
    
    # 记录单模态mAP历史
    single_modal_mAPs_history = {
        'SK': [],
        'NIR': [],
        'CP': [],
        'TEXT': []
    }

    # 梯度累积相关变量（跨epoch累积）
    gradient_accumulation_steps = args.gradient_accumulation_steps
    accumulation_count = 0
    

    # 根据精度类型决定是否使用scaler
    autocast_dtype = args.autocast_dtype
    use_scaler = autocast_dtype == torch.float16
    scaler = GradScaler() if use_scaler else None

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
            
            # 只在累积开始时清零梯度
            if accumulation_count == 0:
                optimizer.zero_grad()

            ret = model(batch,scaler)

            total_loss = sum([v for k, v in ret.items() if "loss" in k]) # 计算损失函数 multi_modal_contrastive_loss损失在模型中计算好了, 并且已经成功detach
            
            # 梯度累积：将损失除以累积步数
            # total_loss = total_loss / gradient_accumulation_steps

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
            
            accumulation_count += 1
            
            # 只在累积步数达到时执行optimizer.step()
            if accumulation_count >= gradient_accumulation_steps:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                accumulation_count = 0

            synchronize() # 分布式计算相关

            
            if (n_iter + 1) % log_period == 0:
                eval_count += 1
                train_loss_list.append(meters['loss'].avg)
                lr_list.append(scheduler.get_lr()[0])
                eval_iters_list.append(eval_count)
                
                # 记录各种损失
                # 注意：meters中的key和loss_dict中的key要对应
                loss_mapping = {
                    'multi_modal_contrastive_sdm_loss': 'multi_modal_contrastive_sdm_loss',
                    'sk_sdm_loss': 'sk_sdm_loss',
                    'nir_sdm_loss': 'nir_sdm_loss', 
                    'cp_sdm_loss': 'cp_sdm_loss',
                    'text_sdm_loss': 'text_sdm_loss',
                    'multi_modal_contrastive_itc_loss': 'multi_modal_contrastive_itc_loss',
                    'sk_itc_loss': 'sk_itc_loss',
                    'nir_itc_loss': 'nir_itc_loss',
                    'cp_itc_loss': 'cp_itc_loss',
                    'text_itc_loss': 'text_itc_loss',
                    'sdm_loss': 'sdm_loss',
                    'itc_loss': 'itc_loss',
                    'id_loss': 'id_loss',
                    'mlm_loss': 'mlm_loss'
                }
                
                for loss_name, meter_key in loss_mapping.items():
                    if meter_key in meters and meters[meter_key].avg > 0:
                        loss_dict[loss_name].append(meters[meter_key].avg)
                    else:
                        loss_dict[loss_name].append(0)
                
                plot_and_save_curves(args.output_dir, len(train_loader), train_loss_list, mAP_list, lr_list, log_period, eval_iters_list, eval_epoch_list, loss_dict, single_modal_mAPs_history)

                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
                wandb.log({"message": info_str})
            
            # 只在实际执行optimizer.step()时调度学习率
            if  (n_iter + 1) % (scheduler_period) == 0:
                scheduler.step(scheduler_period          )# // gradient_accumulation_steps)
                # print(f"Epoch {epoch}, Iteration {n_iter + 1}, Lr: {scheduler.get_lr()[0]:.2e}")
            
        # 跨epoch累积模式：不在epoch结束时强制执行optimizer.step()
        # accumulation_count 保持跨epoch状态
        
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
            wandb.log({"message": f"Epoch {epoch} done. Time per batch: {time_per_batch:.3f}[s] Speed: {train_loader.batch_size / time_per_batch:.1f}[samples/s]"})
        # evalue
        if epoch % eval_period == 0 and args.test_size > 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                wandb.log({"message": f"Validation Results - Epoch: {epoch}"})
                if args.distributed:
                    r1, mAP, single_modal_mAPs = evaluator.eval(model.module.eval())
                else:
                    r1, mAP, single_modal_mAPs = evaluator.eval(model.eval())
                
                eval_epoch_list.append(epoch)
                mAP_list.append(mAP)
                
                # 记录单模态mAP
                for modal_name in ['SK', 'NIR', 'CP', 'TEXT']:
                    if modal_name in single_modal_mAPs:
                        single_modal_mAPs_history[modal_name].append(single_modal_mAPs[modal_name])
                    else:
                        single_modal_mAPs_history[modal_name].append(0)
                
                plot_and_save_curves(args.output_dir, len(train_loader), train_loss_list, mAP_list, lr_list, log_period, eval_iters_list, eval_epoch_list, loss_dict, single_modal_mAPs_history)

                torch.cuda.empty_cache()
                if best_mAP < mAP:
                    best_mAP = mAP
                    arguments["best_mAP_epoch"] = epoch
                    checkpointer.save("best", **arguments)
                logger.info(f"best mAP: {best_mAP} at epoch {arguments['best_mAP_epoch']}")
                wandb.log({"message": f"best mAP: {best_mAP} at epoch {arguments['best_mAP_epoch']}"})
                
        if epoch == num_epoch:
            arguments["best_mAP_epoch"] = epoch
            checkpointer.save("best", **arguments)
            wandb.log({"message": "save success"})

def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger(f"IRRA.test:")
    logger.info("Enter inferencing")
    evaluator = Evaluator_OR(test_img_loader, test_txt_loader)

    return evaluator.eval(model.eval())
