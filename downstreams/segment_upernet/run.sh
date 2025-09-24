# if you want to run all datasets
SCRIPT="/data/zxq/DermINO/downstreams/segment_upernet/train_segmentor.py"
# datasets=("skin_cancer" "ph2_data" "seg_2016" "seg_2017" "seg_2018")
datasets=("skin_cancer")
for DATASET in "${datasets[@]}"; do
    echo "==============================="
    echo "Starting tasks for dataset: $DATASET"
    echo "==============================="

    CUDA_VISIBLE_DEVICES=0 python $SCRIPT \
        --dataset-type "${DATASET}" 

done

