datasets=("derm7pt_data")
# datasets=("med_node"  "ph2_data" "fit_data_2" "fit_data_9" "ddi_data" "derm7pt_data" "acne04_data" "fit_data_1" )
export CUDA_VISIBLE_DEVICES=0

for ds in "${datasets[@]}"; do
    echo "==== Running dataset: $ds ===="
    python /data/zxq/nature/DermINO/pretrain/dinov2/eval/linear.py \
        --batch-size 256 \
        --epochs 50 \
        --dataset-type "$ds"
done
