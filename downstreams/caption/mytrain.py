import os
import argparse
import math
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import numpy as np
import json
import wandb
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from dataset import ImageCaptionDataset, get_train_transform, get_val_transform
from model import ImageCaptionModel

import torchvision.models as models
import torch.nn as nn

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

from types import SimpleNamespace
from peft import get_peft_model, LoraConfig, TaskType
import sys 
sys.path.append("/data/zxq/DermINO/downstreams/caption")
sys.path.append("/data/zxq/DermINO/pretrain")
from dinov2.eval.setup import setup_and_build_model

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Dinov2ImageEncoder(nn.Module):
    def __init__(self, encoder_model, feature_dim=768):
        super().__init__()
        self.encoder = encoder_model
        self.num_features = feature_dim  
    
    def forward(self, x):
        features = self.encoder(x.float())    
        features = features.view(features.size(0), -1) 
        return features.bfloat16()  # 转换为 bfloat16 格式以节省显存
    
class ResNet18ImageEncoder(nn.Module):
    def __init__(self, pretrained=True, feature_dim=512):
        super().__init__()
        # Load ResNet18 without the final classification layer
        resnet = models.resnet18(pretrained=pretrained)
        # Remove the final fully connected layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.num_features = feature_dim
    
    def forward(self, x):
        features = self.encoder(x.float())
        # ResNet outputs features of shape [batch_size, channels, 1, 1]
        # We need to flatten this to [batch_size, channels]
        features = features.squeeze(-1).squeeze(-1)
        return features.bfloat16()  # 转换为 bfloat16 格式以节省显存
    
def compute_bleu(references, hypotheses, n=1):
    """
    利用 nltk 计算 BLEU-n score。
    references: List[str]，参考 caption。
    hypotheses: List[str]，生成的 caption。
    n: BLEU-n, n=1,2,3,4
    """
    # corpus_bleu 要求参考答案为 list of list of tokens
    tokenized_refs = [[ref.split()] for ref in references]
    tokenized_hyps = [hyp.split() for hyp in hypotheses]
    
    if n == 1:
        weights = (1, 0, 0, 0)
    elif n == 2:
        weights = (0.5, 0.5, 0, 0)
    elif n == 3:
        weights = (0.33, 0.33, 0.33, 0)
    elif n == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    else:
        raise ValueError("n must be 1, 2, 3, or 4")
    
    score = corpus_bleu(tokenized_refs, tokenized_hyps, weights=weights)
    return score

def compute_meteor(references, hypotheses):
    """
    计算 METEOR 评分
    """
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = word_tokenize(ref)
        hyp_tokens = word_tokenize(hyp)
        score = meteor_score([ref_tokens], hyp_tokens)
        scores.append(score)
    return sum(scores) / len(scores)

def compute_rouge(references, hypotheses):
    """
    计算 ROUGE 评分
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores),
        'rouge2': sum(rouge2_scores) / len(rouge2_scores),
        'rougeL': sum(rougeL_scores) / len(rougeL_scores)
    }

def compute_bert_score(references, hypotheses):
    """
    计算 BERTScore
    """
    P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False)
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

def compute_cider(references, hypotheses):
    """
    计算 CIDEr 评分
    """
    # CIDEr 要求格式为 {id: [caption]}
    refs = {i: [r] for i, r in enumerate(references)}
    hyps = {i: [h] for i, h in enumerate(hypotheses)}
    
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(refs, hyps)
    return score

def print_gpu_usage(note=""):
    print(f"\n[GPU MEMORY] {note}")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(i) / 1024**2    # MB
        print(f" - GPU {i}: Allocated = {allocated:.2f} MB, Reserved = {reserved:.2f} MB")

def set_seed(seed):
    """Set random seed for reproducibility across all libraries used"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def print_model_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    print("\nDetailed breakdown:")
    for name, param in model.named_parameters():
        print(f"{name:60} | shape: {param.shape} | trainable: {param.requires_grad}")

def train(args, new_args):
    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化本地日志保存（不再使用 wandb）
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train_log.json")
    # 如果已有日志文件，可加载已有日志，否则创建一个空列表
    # with open(log_file, "w") as f:
    #     logs = json.load(f)
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    # 构造数据集
    train_transform = get_train_transform(image_size=args.image_size)
    val_transform = get_val_transform(image_size=args.image_size)
    
    train_dataset = ImageCaptionDataset(args.train_csv, transform=train_transform)
    val_dataset = ImageCaptionDataset(args.val_csv, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    image_encoder, autocast_dtype = setup_and_build_model(new_args)

    
    # 加载语言模型与 tokenizer
    language_model = AutoModelForCausalLM.from_pretrained(
        args.language_model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    #********************new code, add lora************************
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=16,  # LoRA 缩放因子
        lora_dropout=0.05,  # LoRA dropout
        bias="none",  # 不做 bias 的 LoRA
        task_type=TaskType.CAUSAL_LM,  # 任务类型是 Causal Language Modeling
        target_modules=["q_proj", "v_proj"]  # 这应该是 Mistral 模型中的 Attention 层的模块名称
    )

    # 增加冻结大语言模型
    for param in language_model.parameters():
        param.requires_grad = False

    # 为模型添加 LoRA 适配器
    language_model = get_peft_model(language_model, lora_config)

    # 打印可训练参数，确保只有 LoRA 参数是可训练的
    language_model.print_trainable_parameters()
    #********************end*****************************************

    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name)
    eos_token_id = tokenizer.eos_token_id
    
    prefix_text = "<s>[INST] "
    suffix_text = " Please descript this dermatology image: [/INST]"
    prefix_tokens = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt").input_ids
    suffix_tokens = tokenizer(suffix_text, add_special_tokens=False, return_tensors="pt").input_ids
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    language_model.to(device)
    
    # print CUDA
    print_gpu_usage("print CUDA")

    projection_hidden_dim = language_model.config.hidden_size
    #**************new code*************
    get_feature_dim=768
    #**********************************
    model = ImageCaptionModel(image_encoder, language_model, prefix_tokens, suffix_tokens, projection_hidden_dim, args.num_image_tokens, get_feature_dim=get_feature_dim)
    model.to(device)
    
    print_model_parameters(model)

    optimizer = optim.AdamW([
        {'params': model.projection.parameters(), 'lr': args.adapter_lr},
        {'params': model.language_model.parameters(), 'lr': args.llm_lr}
    ], weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_ratio = 0.1
    warmup_steps = int(warmup_ratio * total_steps)


    for epoch in range(args.epochs):
        print("start train")
        model.train()
        train_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, captions in pbar:
            images = images.to(device).bfloat16()
            tokenized = tokenizer(captions, return_tensors="pt", padding=True, padding_side="right", truncation=True, max_length=256)
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)
            
            outputs = model(images, input_ids, attention_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        

        avg_train_loss = train_loss / num_batches
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        references = []
        hypotheses = []
        with torch.no_grad():
            for images, captions in tqdm(val_loader, desc="Validating"):
                images = images.to(device).bfloat16()
                tokenized = tokenizer(captions, return_tensors="pt", padding=True, padding_side="right", truncation=True, max_length=256)
                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)
                
                outputs = model(images, input_ids, attention_mask)
                loss = outputs.loss
                val_loss += loss.item()
                num_val_batches += 1
                
                generated_ids = model.generate(
                    images,
                    max_new_tokens=args.max_gen_length,
                    num_beams=1,
                    do_sample=False,  
                    eos_token_id=eos_token_id,
                    early_stopping=True,
                )
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                references.extend([x[:-5] for x in captions])
                hypotheses.extend(generated_texts)
            
            avg_val_loss = val_loss / num_val_batches
            perplexity = math.exp(avg_val_loss) if avg_val_loss < 700 else float('inf')
            
            bleu1 = compute_bleu(references, hypotheses, n=1)
            bleu2 = compute_bleu(references, hypotheses, n=2)
            bleu4 = compute_bleu(references, hypotheses, n=4)
            meteor = compute_meteor(references, hypotheses)
            rouge_scores = compute_rouge(references, hypotheses)
            bert_scores = compute_bert_score(references, hypotheses)
            cider = compute_cider(references, hypotheses)
            
            metrics_dict = {
                "epoch": epoch+1,
                "val_loss": avg_val_loss,
                "perplexity": perplexity,
                "BLEU-1": bleu1,
                "BLEU-2": bleu2,
                "BLEU-4": bleu4,
                "METEOR": meteor,
                "CIDEr": cider,
                "ROUGE-1": rouge_scores['rouge1'],
                "ROUGE-2": rouge_scores['rouge2'],
                "ROUGE-L": rouge_scores['rougeL'],
                "BERTScore_precision": bert_scores['precision'],
                "BERTScore_recall": bert_scores['recall'],
                "BERTScore_f1": bert_scores['f1']
            }
            

            # 将指标写入日志列表，然后保存为 JSON 文件
            logs.append(metrics_dict)
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=2)
            
            print(f"Epoch {epoch+1} Evaluation Metrics:")
            for key, value in metrics_dict.items():
                if key != "epoch":
                    print(f"{key}: {value:.4f}")
        
        # 保存模型 checkpoint（如需要）
        if args.save_model:
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='模型训练脚本')
    parser.add_argument('--epochs',
                    type=int,
                    required=False,
                    help='训练的轮数（默认为10）')

    parser.add_argument('--output_dir',
                        type=str,
                        required=False,
                        help='模型输出保存目录（默认为 ./outputs）')

    parser.add_argument('--config_file',
                        type=str,
                        required=False,
                        help='config_file')

    parser.add_argument('--pretrained_weights',
                        type=str,
                        required=False,
                        help='pretrained_weights')

    parser.epochs = 2
    parser.output_dir = '/data/zxq/DermINO/downstreams/caption/outputs/'
    parser.config_file = '/data/zxq/DermINO/checkpoint/dermino/config.yaml'
    parser.pretrained_weights = '/data/zxq/DermINO/checkpoint/dermino/teacher_checkpoint.pth'

    print(f"训练时长: {parser.epochs}")
    print(f"输出路径: {parser.output_dir}")
    print(f"训练时长: {parser.config_file}")
    print(f"输出路径: {parser.pretrained_weights}")
    
    batch_size = 4
    print(f"batch_size: {batch_size}")


    output_dir = parser.output_dir
    args = SimpleNamespace(
        seed=114514,
        train_csv="/data/zxq/DermINO/extra_files/new_train_3600_dataset.csv",
        val_csv="/data/zxq/DermINO/extra_files/new_test_400_dataset.csv",
        image_size=224,
        batch_size=batch_size,
        epochs=parser.epochs,
        adapter_lr=4e-4,
        weight_decay=0.05,
        llm_lr=8e-5,
        num_image_tokens=8,
        image_feature_dim=512,
        language_model_name="/data/zxq/DermINO/checkpoint/biomistral/",
        max_gen_length=256,
        output_dir=output_dir,
        save_model=False,
        wandb_project="ImageCaptionProject",  # 该参数现已不使用，可保留备用
    )

    new_args = argparse.Namespace(
        config_file="/data/zxq/DermINO/checkpoint/dermino/config.yaml",
        output_dir=output_dir,
        pretrained_weights="/data/zxq/DermINO/checkpoint/dermino/teacher_checkpoint.pth",
        opts=[],
    )


    train(args, new_args)