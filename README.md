# DermINO: Hybrid Pretraining for a Versatile Dermatology Foundation Model 

**Abstract:** We present DermINO, a versatile foundation model for dermatology. Trained on a curated dataset of 432,776 images from three sources (public repositories, web-sourced images, and proprietary collections), DermINO incorporates a novel hybrid pretraining framework that augments the self-supervised learning paradigm through semi-supervised learning and knowledge-guided prototype initialization. 
This integrated method not only deepens the understanding of complex dermatological conditions, but also substantially enhances the generalization capability across various clinical tasks. 
Evaluated across 20 datasets, DermINO consistently outperforms state-of-the-art models across a wide range of tasks. It excels in high-level clinical applications including malignancy classification, disease severity grading, multi-category diagnosis, and dermatological image caption, while also achieving state-of-the-art performance in low-level tasks such as skin lesion segmentation. Furthermore, DermINO demonstrates strong robustness in privacy-preserving federated learning scenarios and across diverse skin types and sexes. 


## Updates
- 06/2026: The ViT-base version of Dermino is now available.

## 1. Download Dermino Pre-trained Weights

### Obtaining Model Weights

| Model Name | Release Date | Model Architecture | Google Drive Link/HuggingFace | 
|:------------:|:------------:|:-------------------:|:------------------:|
| Dermino | 06/2026 | ViT-B/14 | [Link](https://drive.google.com/file/d/1-v8N6LNISdcF7SGYpTNq5W9szoiyZGRJ/view?usp=drive_link) | 


## 2. Data Preparation

### Public Dataset Links and Splits

| Pretrain Dataset | Dataset link |
|---------|---------------|
| CSID-CJFH | Proprietary |
| ISIC-Duplicate ||
| SCIN | [link](https://github.com/google-research-datasets/scin) |
| SD-198 | [link](https://huggingface.co/datasets/resyhgerwshshgdfghsdfgh/SD-198) |
| Dermnet | [link](https://dermnet.com/) |
| PAD-UFES-20 | [link](https://data.mendeley.com/datasets/zr7vgbcyr2/1) |
| LESION130K | Web |
| Downstream Dataset | Dataset link |
| MPL5 | Proprietary |
| DDI | [link](https://ddi-dataset.github.io/) |
| Fitzpatrick17k-2 | [link](https://github.com/mattgroh/fitzpatrick17k) |
| MED-Node | [link](https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/) |
| PH2(cls) | [link](https://www.fc.up.pt/addi/ph2%20database.html) |
| MSD2 | Proprietary |
| MTD2 | Proprietary |
| ACNE04 | [link](https://github.com/xpwu95/LDL) |
| GLD6 | Proprietary |
| SID2 | Proprietary |
| VWCD4 | Proprietary |
| Derm7pt | [link](https://github.com/jeremykawahara/derm7pt) |
| Fitzpatrick17k-3 | [link](https://github.com/mattgroh/fitzpatrick17k) |
| Fitzpatrick17k-9 | [link](https://github.com/mattgroh/fitzpatrick17k) |
| SkinCAP | [link](https://huggingface.co/datasets/joshuachou/SkinCAP) |
| PH2(seg) | [link](https://www.fc.up.pt/addi/ph2%20database.html) |
| Skincancer | [link](https://vip.uwaterloo.ca/) |
| ISIC2016 | [link](https://challenge.isic-archive.com/landing/2016/) |
| ISIC2017 | [link](https://challenge.isic-archive.com/landing/2017/) |
| ISIC2018 | [link](https://challenge.isic-archive.com/landing/2018/) |


## Acknowlegdement

This code is built on [DINOv2](https://github.com/facebookresearch/dinov2). We thank the authors for sharing their code.


## Citation
```bibtex
@article{xu2025dermino,
  title={DermINO: Hybrid Pretraining for a Versatile Dermatology Foundation Model},
  author={Xu, Jingkai and Cheng, De and Zhao, Xiangqian and Yang, Jungang and Wang, Zilong and Jiang, Xinyang and Luo, Xufang and Chen, Lili and Ning, Xiaoli and Li, Chengxu and others},
  journal={arXiv preprint arXiv:2508.12190},
  year={2025}
}
```
