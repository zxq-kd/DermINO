# CUDA_VISIBLE_DEVICES=1  python /data/zxq/DermINO/pretrain/DermINO_train.py \
# train.dataset_path="MyDataset:root=/data/zxq/nature/nature_data/compare_data/fit_data/benign" \

# run for pretrain
torchrun --nproc_per_node 8 /data/zxq/DermINO/pretrain/DermINO_train.py \
    --config-file /data/zxq/DermINO/pretrain/dinov2/configs/train/DermINO.yaml \
    --output-dir  /data/zxq/DermINO/pretrain/output_dir  \
    train.dataset_path="MyDataset:root=/data/zxq/DermINO/datasets/pretrain" \
    train.class_weight=1 \
    train.OFFICIAL_EPOCH_LENGTH=400 \
    train.batch_size_per_gpu=256 \
    optim.base_lr=2e-3 \
    optim.layerwise_decay=1  \
    optim.patch_embed_lr_mult=0.2  \
    optim.epochs=100 \
    optim.warmup_epochs=10  \
    student.pretrained_weights="/data/zxq/DermINO/checkpoint/dinov2_vitb14_resize.pth" \
    student.arch=vit_base \
    student.patch_size=14 \
    student.ffn_layer="mlp" \
    crops.local_crops_size=98 \
    crops.global_crops_size=224 \
    teacher.momentum_teacher=0.992 





