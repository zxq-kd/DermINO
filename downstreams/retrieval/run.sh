# datasets=("med_node"  "ph2_data" "fit_data_2" "fit_data_9" "ddi_data" "derm7pt_data" "acne04_data" "fit_data_1" )
datasets=("derm7pt_data")
export CUDA_VISIBLE_DEVICES=0

for ds in "${datasets[@]}"; do
    echo "==== Running dataset: $ds ===="
    python /data/zxq/nature/DermINO/pretrain/dinov2/eval/knn.py \
        --dataset-type "$ds"
done
