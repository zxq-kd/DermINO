# DermINO: Hybrid Pretraining for a Versatile Dermatology Foundation Model 

**Abstract:** We present DermINO, a versatile foundation model for dermatology. Trained on a curated dataset of 432,776 images from three sources (public repositories, web-sourced images, and proprietary collections), DermINO incorporates a novel hybrid pretraining framework that augments the self-supervised learning paradigm through semi-supervised learning and knowledge-guided prototype initialization. 
This integrated method not only deepens the understanding of complex dermatological conditions, but also substantially enhances the generalization capability across various clinical tasks. 
Evaluated across 20 datasets, DermINO consistently outperforms state-of-the-art models across a wide range of tasks. It excels in high-level clinical applications including malignancy classification, disease severity grading, multi-category diagnosis, and dermatological image caption, while also achieving state-of-the-art performance in low-level tasks such as skin lesion segmentation. Furthermore, DermINO demonstrates strong robustness in privacy-preserving federated learning scenarios and across diverse skin types and sexes. 

## Installation
First, clone the repo and cd into the directory:
```shell
git clone https://github.com/SiyuanYan1/PanDerm
cd PanDerm/classification
```
Then create a conda env and install the dependencies:
```shell
conda create -n PanDerm python=3.10 -y
conda activate PanDerm
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
cd classification
pip install -r requirements.txt
```

## 1. Download PanDerm Pre-trained Weights

### Obtaining Model Weights

| Model Name | Release Date | Model Architecture | Google Drive Link | 
|:------------:|:------------:|:-------------------:|:------------------:|
| PanDerm_Base | 04/2025 | ViT-B/16 | [Link](https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB/view?usp=sharing) | 
| PanDerm (proposed in our paper) | 10/2024 | ViT-L/16 | [Link](https://drive.google.com/file/d/1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE/view?usp=sharing) |

## 2. Data Preparation

<details>
  <summary>
  <b>Using Your Own Dataset</b>
  </summary>
  
If you wish to use our model with your own dataset, the dataset used for linear probing or finetuning should be organized in a CSV file with the following structure:

**Required Columns**
- `image`: Path to the image file (e.g., ISIC_0034524.jpg)
- `split`: Dataset partition indicator (train, val, or test)
- For multi-class classification:
  - `label`: Numerical class label (e.g., 0, 1, 2, 3, 4)
- For binary classification:
  - `binary_label`: Binary class label (e.g., 0, 1)

For Multi-class Example:
```csv
image,label,split
ISIC_0034524.jpg,1,train
ISIC_0034525.jpg,1,train
ISIC_0034526.jpg,4,val
ISIC_0034527.jpg,3,test
```

For Binary Classification Example:
```csv
image,binary_label,split
ISIC_0034524.jpg,1,train
ISIC_0034525.jpg,1,train
ISIC_0034526.jpg,0,val
ISIC_0034527.jpg,0,test
```
</details>

<details>
    <summary>
<b> Using Pre-processed Public Datasets </b>
    </summary>


We've already pre-processed several public datasets to reproduce the results in our study and prevent data leakage between splits. These datasets are ready to use with our model and require no additional formatting.

### Public Dataset Links and Splits

| Dataset | Processed Data | Original Data |
|---------|----------------|---------------|
| HAM10000 | [Download](https://drive.google.com/file/d/1D9Q4B50Z5tyj5fd5EE9QWmFrg66vGvfA/view?usp=sharing) | [Official Website](https://challenge.isic-archive.com/data/#2018) |
| BCN20000 | [Download](https://drive.google.com/file/d/1jn1h1jWjd4go7BQ5fFWMRBMtq7poSlfi/view?usp=sharing) | [Official Website](https://figshare.com/articles/journal_contribution/BCN20000_Dermoscopic_Lesions_in_the_Wild/24140028/1) |
| DDI | [Download](https://drive.google.com/file/d/1F5RVqBUIxYcub1OkBm6yHTyV2TkHc65B/view?usp=sharing) | [Official Website](https://ddi-dataset.github.io/index.html) |
| Derm7pt | [Download](https://drive.google.com/file/d/1OYAmqG93eWLdf7dIkulY_fr0ZScvRLRg/view?usp=sharing) | [Official Website](https://derm.cs.sfu.ca/Welcome.html) |
| Dermnet | [Download](https://drive.google.com/file/d/1WrvReon2gA3sF9rqQGqivglG7HLFJ8he/view?usp=sharing) | [Official Website](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) |
| HIBA | [Download](https://drive.google.com/file/d/1Sg0gFhfBaNNoeunF7C0HZgDbp5EDV436/view?usp=sharing) | [Official Website](https://www.isic-archive.com) |
| MSKCC | [Download](https://drive.google.com/file/d/17ma4tREXHAq1ZcBT7lZBhwO-3UHSbDW2/view?usp=sharing) | [Official Website](https://www.isic-archive.com) |
| PAD-UFES | [Download](https://drive.google.com/file/d/1NLv0EH3QENuRxW-_-BSf4KMP9cPjBk9o/view?usp=sharing) | [Official Website](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer) |
| PATCH16 | [Download](https://drive.google.com/file/d/1wDMIfYrQatkeADoneHgjXQrawVMK-TFL/view?usp=sharing) | [Official Website](https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/7QCR8S) |

**Note:** The processed datasets provided here may differ slightly from those on the official websites. To ensure reproducibility of our paper's results, please use the processed data links above.
</details>

## 3. Linear Evaluation on Image Classification Tasks

Training and evaluation using the PAD-UFES dataset as an example. Replace the CSV path and root path with your own dataset.

### Key Parameters
- `batch_size`: Adjust based on the memory size of your GPU.
- `model`: Model size - "PanDerm_Large_LP" (original paper model) or "PanDerm_Base_LP" (smaller version)
- `nb_classes`: Set this to the number of classes in your evaluation dataset.
- `percent_data`: Controls the percentage of training data used. For example, 0.1 means evaluate models using 10% of the training data. Modify this if you want to conduct label efficiency generalization experiments.
- `csv_path`: Organize your dataset as described in the "Data Preparation" section.
- `root_path`: The path of your folder for saved images. 
- `pretrained_checkpoint`: Path to the pretrain checkpoint - "panderm_ll_data6_checkpoint-499.pth" for "PanDerm_Large_LP" and "panderm_bb_data6_checkpoint-499.pth" for "PanDerm_Base_LP".
  
### Evaluation Command

```bash
cd classification
CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm_Large_LP" \
  --nb_classes 6 \
  --percent_data 1.0 \
  --csv_filename "PanDerm_Large_LP_result.csv" \
  --output_dir "/path/to/your/PanDerm/output_dir/PanDerm_res/" \
  --csv_path "/path/to/your/PanDerm/Evaluation_datasets/pad-ufes/2000.csv" \
  --root_path "/path/to/your/PanDerm/Evaluation_datasets/pad-ufes/images/ " \
  --pretrained_checkpoint "/path/to/your/PanDerm/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"
```

### More Usage Cases

For additional evaluation datasets, please refer to the bash scripts for detailed usage. We provide running code to evaluate on 9 public datasets. You can choose the model from the available options.

To run the evaluations:

```bash
cd classification
bash script/lp_reproduce.sh
```

## 4. Fine-tuning on Image Classification Tasks

### Key Parameters

- `model`: Model size - "PanDerm_Large_FT" (original paper model) or "PanDerm_Base_FT" (smaller version)
- `pretrained_checkpoint`: Path to the pretrain checkpoint - "panderm_ll_data6_checkpoint-499.pth" for "PanDerm_Large_FT" and "panderm_bb_data6_checkpoint-499.pth" for "PanDerm_Base_FT".
- `nb_classes`: Set this to the number of classes in your evaluation dataset.
-  `weights`: Setting to use the weighted random sampler for the imbalanced class dataset.
-  `monitor`: Choosing your checkpoint based on "acc" or "recall".
- `csv_path`: Organize your dataset as described in the "Data Preparation" section.
- `root_path`: The path of your folder for saved images. 
-  `TTA`: Enable Test-Time Augmentation. You can modify the augmentation setting in the class `TTAHandler` [classification/furnace/engine_for_finetuning.py](classification/furnace/engine_for_finetuning.py).
-- `eval`: Model inference.

### Recommended Configuration for fine-tuning

Our experiments show the following hyperparameters deliver optimal performance across various evaluation datasets:
- Batch size: 128
- Learning rate: 5e-4
- Training epochs: 50
- Enable the weighted random sampler
- Enable TTA during testing 
  
We observed that the hyperparameter setting is robust across datasets and typically doesn't require adjustment.

### Start Training

You could fine-tune PanDerm on your dataset. Here is a command-line example for fine-tuning PanDerm_Large on the PAD-UFES dataset:


```bash
MODEL_NAME="PanDerm_Large_FT"
MODEL_PATH="/path/to/your/PanDerm/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"

CUDA_VISIBLE_DEVICES=0 python3 run_class_finetuning.py \
    --model $MODEL_NAME \
    --pretrained_checkpoint $MODEL_PATH \
    --nb_classes 6 \
    --batch_size 128 \
    --lr 5e-4 \
    --update_freq 1 \
    --warmup_epochs 10 \
    --epochs 50 --layer_decay 0.65 --drop_path 0.2 \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 \
    --weights \
    --sin_pos_emb \
    --no_auto_resume \
    --exp_name exp_name "pad finetune and eval" \
    --imagenet_default_mean_and_std \
    --wandb_name "Reproduce_PAD_FT_${seed}" \
    --output_dir /path/to/your/PanDerm/Evaluation_datasets/PAD_Res/ \ # Your best epoch's fine-tuned checkpoint and model output results on the test set will be saved in this directory
    --csv_path "/path/to/your/PanDerm/Evaluation_datasets/pad-ufes/2000.csv" \
    --root_path "/path/to/your/PanDerm/Evaluation_datasets/pad-ufes/images/ " \
    --seed 0 
```

The script for fine-tuning and evaluating PanDerm:

```bash
cd classification
bash script/finetune_train.sh 
```
Note: Remember to adjust the `pretrained_checkpoint` argument to your storage location of pretrained model weights.

### Evaluation
```bash
cd classification
bash script/finetune_test.sh
```
Note: Remember to adjust the `resume` argument to your storage location of finetuned model weights.

## 5. Skin Lesion Segmentation

Please refer to details [here](Segmentation.md).

## License
The model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial academic research purposes with proper attribution.

## Acknowlegdement

This code is built on [CAEv2](https://github.com/Atten4Vis/CAE), [UNI](https://github.com/mahmoodlab/UNI), [MAE](https://github.com/facebookresearch/mae). We thank the authors for sharing their code.

## Citation
```bibtex
@article{yan2025multimodal,
  title={A multimodal vision foundation model for clinical dermatology},
  author={Yan, Siyuan and Yu, Zhen and Primiero, Clare and Vico-Alonso, Cristina and Wang, Zhonghua and Yang, Litao and Tschandl, Philipp and Hu, Ming and Ju, Lie and Tan, Gin and others},
  journal={Nature Medicine},
  pages={1--12},
  year={2025},
  publisher={Nature Publishing Group}
}
```
