DATASET_NAME="ORBench"

CUDA_VISIBLE_DEVICES=5 \
python train.py \
--batch_size 24 \
--loss_name 'multi_modal_contrastive+itc' \
--sampler random \
--pretrain_choice '/SSD_Data01/zyl/prcv_work/model_cache/huggingface_model/model.safetensors' \
--test_size 0.125 \
--eval_period 20 \
--drop_last 1 \
--img_aug \
--MLM \
--dataset_name $DATASET_NAME \
--name fgclip \
--root_dir '/SSD_Data01/PRCV-ReID5o/data/' \
--num_epoch 800 \
--lr 5e-6 \
--warmup_epochs 580 \
--lrscheduler exp \
--power 0.5 \
--step_size 2000 \
--img_size 224,224 \
--freeze_embedding_layers \
--lr_moe_frm 1e-1 \
--moe_num_experts 64 \
--moe_top_k 3 \
--moe_aux_loss_weight 100


"""
--freeze_embedding_layers \
--freeze_projection_layers \
--add_multimodal_embeddings \
--add_multimodal_projections \

--use_multimodal_layers_in_pairs \
"""

