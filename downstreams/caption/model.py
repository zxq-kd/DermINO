import torch
import torch.nn as nn

class ImageCaptionModel(nn.Module):
    """
    构建类似 LLava 的多模态模型：
    1. 图像经过预训练 image encoder（参数冻结）得到特征向量；
    2. 特征向量经过一个 projection 层映射为一段固定长度的 token 序列（这里将 image feature 映射为 [num_image_tokens x hidden_dim]）；
    3. 将 projection 得到的图像 tokens 与文本 token（由 caption tokenizer 得到）进行拼接，
       然后送入语言模型进行文本生成以及计算语言模型的损失（前缀部分不计算 loss）。
    """
    def __init__(self, image_encoder, language_model, prefix_tokens, suffix_tokens, projection_hidden_dim, num_image_tokens, get_feature_dim=None):
        """
        image_encoder: 预训练的图像编码器，要求有 forward() 方法和 feature_dim 属性（输出向量维度）。
        language_model: 预训练的语言模型（如 BioMistrial-7B），要求支持 inputs_embeds 输入；
        projection_hidden_dim: 对应语言模型的 hidden size。
        num_image_tokens: 将图像特征扩展成多少个“虚拟” token。
        """
        super().__init__()
        self.image_encoder = image_encoder
        # 冻结图像编码器参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.num_image_tokens = num_image_tokens
        self.language_model = language_model
        self.prefix_tokens = prefix_tokens
        self.suffix_tokens = suffix_tokens
        
        image_feature_dim = get_feature_dim


        self.projection_hidden_dim = projection_hidden_dim
        self.projection = nn.Linear(image_feature_dim, num_image_tokens * projection_hidden_dim, dtype=torch.bfloat16)
        
        # 预处理并缓存指令token
        prefix_text = "<s> [INST] "
        suffix_text = " Please descript this dermatology image: [/INST]"

        self.prefix_len = self.prefix_tokens.size(1)
        self.suffix_len = self.suffix_tokens.size(1)
    
    def forward(self, images, input_ids, attention_mask=None):
        """
        images: Tensor，shape [batch_size, channels, height, width]
        input_ids: Tensor，shape [batch_size, seq_len]，caption 的 token 化 id
        attention_mask: Tensor，shape [batch_size, seq_len]（可选）
        """
        batch_size = images.size(0)
        # 得到图像特征，假设 shape 为 [batch_size, feature_dim]
        # *********************new code ************************
        # image_features = self.image_encoder(images)
        with torch.no_grad():
            image_features = self.image_encoder(images.float()).bfloat16()


        #*********************************************************
        # 投影获得图像的 "虚拟" token 表示
        projected = self.projection(image_features)  # [batch_size, num_image_tokens * hidden_dim]
        projected = projected.view(batch_size, self.num_image_tokens, self.projection_hidden_dim)  # [batch, num_image_tokens, hidden_dim]
        
        
        prefix_tokens = self.prefix_tokens.to(images.device)
        suffix_tokens = self.suffix_tokens.to(images.device)
        
        prefix_embeds = self.language_model.get_input_embeddings()(prefix_tokens)
        suffix_embeds = self.language_model.get_input_embeddings()(suffix_tokens)
        
        prefix_embeds = prefix_embeds.expand(batch_size, -1, -1)    #[batch_size, prefix_len, hidden_dim]
        suffix_embeds = suffix_embeds.expand(batch_size, -1, -1)    #[batch_size, suffix_len, hidden_dim]
        
        # 使用语言模型的 embedding 层将文本 token 转换为 embedding，
        # 这里推荐用 get_input_embeddings() 方法（注意不同模型调用名称可能有所差异）
        text_embeds = self.language_model.get_input_embeddings()(input_ids)  # [batch, seq_len, hidden_dim]
        
        # 拼接：首先是 prefix_embeds, image tokens，suffix_embeds, 再接上文本 embedding
        # combined_embeds = torch.cat([projected, text_embeds], dim=1)  # [batch, num_image_tokens+seq_len, hidden_dim]
        combined_embeds = torch.cat([
            prefix_embeds,       # <s> [INST] 
            projected,           # <image tokens>
            suffix_embeds,       # Please descript this dermatology image: [/INST]
            text_embeds          # 原始caption
        ], dim=1)
        
        
        prefix_mask = torch.ones(batch_size, self.prefix_len, device=images.device) 
        img_mask = torch.ones(batch_size, self.num_image_tokens, device=images.device)
        suffix_mask = torch.ones(batch_size, self.suffix_len, device=images.device)
        
        if attention_mask is None:
            text_mask = torch.ones(input_ids.size(), device=images.device, dtype=torch.long)
        else:
            text_mask = attention_mask
            
        combined_attention_mask = torch.cat([
            prefix_mask, img_mask, suffix_mask, text_mask
        ], dim=1)
        
        prefix_labels = torch.full((batch_size, self.prefix_len), -100, 
                                device=images.device, dtype=input_ids.dtype)
        img_labels = torch.full((batch_size, self.num_image_tokens), -100, 
                                device=images.device, dtype=input_ids.dtype)
        suffix_labels = torch.full((batch_size, self.suffix_len), -100, 
                                device=images.device, dtype=input_ids.dtype)
        if attention_mask is None:
            text_labels =input_ids
        else:
            text_labels = input_ids.masked_fill(attention_mask == 0, -100)
        
        combined_labels = torch.cat([
            prefix_labels, img_labels, suffix_labels, text_labels
        ], dim=1)
        
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels
        )
        return outputs
    
    def generate(self, images, **generate_kwargs):
        batch_size = images.size(0)
        
        with torch.no_grad():
            image_features = self.image_encoder(images.float()).bfloat16()

        projected = self.projection(image_features)
        projected = projected.view(batch_size, self.num_image_tokens, self.projection_hidden_dim)
        
        # 获取指令token嵌入
        prefix_tokens = self.prefix_tokens.to(images.device)
        suffix_tokens = self.suffix_tokens.to(images.device)
        
        prefix_embeds = self.language_model.get_input_embeddings()(prefix_tokens)
        suffix_embeds = self.language_model.get_input_embeddings()(suffix_tokens)
        
        # 扩展到batch size
        prefix_embeds = prefix_embeds.expand(batch_size, -1, -1)
        suffix_embeds = suffix_embeds.expand(batch_size, -1, -1)
        
        prompt_embeds = torch.cat([
            prefix_embeds,       # [INST] 
            projected,           # <image tokens>
            suffix_embeds        # Please descript this dermatology image: [/INST]
        ], dim=1)
        
        total_seq_length = prompt_embeds.size(1)
        attention_mask = torch.ones((batch_size, total_seq_length), device=images.device, dtype=torch.long)
        
        if 'pad_token_id' not in generate_kwargs:
            generate_kwargs['pad_token_id'] = self.language_model.config.eos_token_id
        
        generated_ids = self.language_model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            **generate_kwargs
        )
        return generated_ids