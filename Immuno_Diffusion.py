import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, CLIPTextModel, CLIPTokenizer
from torch_geometric.nn import GATConv
import math
from torch_geometric.nn import global_mean_pool
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from PIL import Image # 用于处理图像输出
from transformers import BertTokenizer

# TODO List:
# --- 整合与验证 ---
# - [DONE] 将模块集成到完整的扩散模型框架 (ImmunoDiffusionModel). 
#   - NOTE: validation_script.py 使用独立的生成循环，未来需考虑将 ImmunoDiffusionModel 应用于验证脚本或对齐两者逻辑。
# - [NEW] validation_script.py: 集成 ImmuneMemoryModule, 将其输出 (memory_signal) 提供给 SEU。
# - [NEW] validation_script.py: 实现真实的 FID 计算, 而非返回占位符。
# - [PARTIALLY DONE] 实现 SLI, ARD 等评估指标 (SLI, ARD 已在 validation_script 中实现, FID 为占位符)。

# --- 核心模块增强 ---
# - [NEW] PDU: 当前 PDU 实现不使用 __init__ 中传入的 sensitive_keywords。需明确其风险检测逻辑 (关键词 vs 预训练分类器)，并对齐 validation_script 中的用法和 README 中的描述。
# - PrivacyEnhancementUnit: 替换为更复杂的、基于Langevin Dynamics的噪声生成或潜在空间扰动机制。
# - PrivacyEnhancementUnit.forward: 结合 memory_signal 实现更复杂的潜在空间扰动。
# - PrivacyDetectionUnit.forward: 可以使用更复杂的图池化方法 (e.g., global_add_pool, attention-based pooling)。
# - [NOT USED] PrivacyEnhancementUnit.shm_update: 当前该方法未被调用，需评估其有效性并决定是否集成。

# --- 高级概念实现 ---
# - [PARTIALLY DONE / PLACEHOLDER] 实现 Apoptosis (注意力自毁), Epigenetic Regulation (提示词加密), Quorum Quenching (分布式)。
# - [PARTIALLY DONE / PLACEHOLDER] ApoptosisMechanism: 实现检测高风险/攻击并禁用模型部分（如UNet中的Attention层）的逻辑。
# - [PARTIALLY DONE / PLACEHOLDER] epigenetic_prompt_encoding: 实现鲁棒的提示词编码/加密机制。
# - ImmunoDiffusionModel: 在 forward 逻辑中添加对 epigenetic_prompt_encoding 的调用。

# --- 已完成/已确认 ---
# - [DONE] ImmuneMemoryModule: 确认 B 的初始化策略是否需要调整。
# - [DONE] ImmuneMemoryModule.update_memory: 考虑更复杂的合并策略。
# - [DONE] ImmuneMemoryModule.update_memory: 限制记忆库大小 (例如 FIFO 或基于重要性采样)。
# - [DONE] ImmuneMemoryModule.query_memory: 定义如何使用查询结果，例如返回最相似的记忆向量或加权平均等。
# - [DONE] ImmunoDiffusionModel.forward: 实现完整的扩散循环，集成PDU, SEU, Memory, Apoptosis。
# - [DONE] ImmunoDiffusionModel.forward: 确保所有组件的输入输出维度匹配。
# - [DONE] ImmunoDiffusionModel.forward: 添加对 device (cuda/cpu) 的处理。

class PrivacyEnhancementUnit(nn.Module):
    """
    安全增强单元 (SEU - Security Enhancement Unit) - Simplified Version
    模拟 B 细胞生成抗体（隐私保护噪声/扰动）的过程。
    在扩散模型的潜在空间中操作，根据风险评估动态调整防御策略。
    """
    def __init__(self, latent_dim, simple_noise_level=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.simple_noise_level = simple_noise_level

    def forward(self, z, t, risk_score, memory_signal=None):
        """
        Args:
            z (torch.Tensor): 当前步的潜在表示，形状 [B, C, H, W]
            t (torch.Tensor): 当前时间步（未使用）
            risk_score (torch.Tensor): 风险评分，形状 [B, 1]
            memory_signal (torch.Tensor or None): 免疫记忆信号，形状 [B, memory_dim]
        Returns:
            torch.Tensor: 添加扰动后的潜变量
        """
        current_risk_score = risk_score.view(-1, 1, 1, 1)  # 形状匹配 z

        base_noise_level = self.simple_noise_level

        if memory_signal is not None:
            # 计算记忆信号强度（范数）
            memory_strength = memory_signal.norm(dim=1, keepdim=True)  # [B, 1]
            memory_strength = memory_strength.view(-1, 1, 1, 1)  # 形状匹配 z

            threshold_high = 1.0
            threshold_low = 0.3

            if (memory_strength > threshold_high).any():
                # 强烈记忆信号，使用非高斯脉冲噪声
                noise = self.generate_targeted_noise(z.shape, device=z.device)
                noise = noise * current_risk_score * base_noise_level * 2.0  # 放大噪声强度
            elif (memory_strength > threshold_low).any():
                # 中等记忆信号，使用加权高斯噪声
                noise = torch.randn_like(z) * current_risk_score * base_noise_level * (1.0 + memory_strength)
            else:
                # 低记忆信号，使用基础高斯噪声
                noise = torch.randn_like(z) * current_risk_score * base_noise_level
        else:
            # 无记忆信号，使用基础高斯噪声
            noise = torch.randn_like(z) * current_risk_score * base_noise_level

        z_perturbed = z + noise
        return z_perturbed

    def generate_targeted_noise(self, shape, device):
        """
        生成非高斯"精确制导"噪声，示例为稀疏脉冲噪声。
        """
        noise = torch.zeros(shape, device=device)
        batch_size = shape[0]
        num_spikes = max(1, batch_size // 10)  # 例如批量大小的十分之一

        for _ in range(num_spikes):
            idx = torch.randint(0, batch_size, (1,))
            noise[idx] = torch.randn_like(noise[idx]) * 5.0  # 放大扰动幅度

        return noise
    
    def adaptive_noise(self, risk_level):
        """
        根据风险等级自适应生成噪声强度和形态。
        Args:
            risk_level (torch.Tensor): 风险等级，范围0~1，形状可为任意可广播形状。
        Returns:
            torch.Tensor: 噪声强度张量，与 risk_level 形状相同。
        """
        # 参数示例：阈值和最大噪声强度
        low_risk_threshold = 0.2
        high_risk_threshold = 0.8
        max_noise_level = self.simple_noise_level  # 最大噪声强度
        min_noise_level = 0.01  # 最小噪声强度，避免完全无噪声

        # 非线性映射：风险低于低阈值时噪声很小，超过高阈值时噪声接近最大
        noise_strength = torch.where(
            risk_level < low_risk_threshold,
            min_noise_level * torch.ones_like(risk_level),
            torch.where(
                risk_level > high_risk_threshold,
                max_noise_level * torch.ones_like(risk_level),
                # 中间区间线性插值
                min_noise_level + (max_noise_level - min_noise_level) * (
                    (risk_level - low_risk_threshold) / (high_risk_threshold - low_risk_threshold)
                )
            )
        )

        # 进一步调整噪声形态，比如根据风险等级调整噪声的分布形状
        # 这里示例用风险等级控制噪声的偏度（skewness）或方差
        # 简单示例：高风险时噪声方差放大1.5倍
        noise_variance_scale = 1.0 + 0.5 * (risk_level - low_risk_threshold).clamp(min=0) / (1 - low_risk_threshold)
        noise_strength = noise_strength * noise_variance_scale

        return noise_strength

    #       def adaptive_noise(self, risk_level): # Removed for simplification
    #           pass上面


    def shm_update(self, feedback_signal=None):
        """
        根据反馈信号动态调整噪声强度参数 simple_noise_level。
        Args:
            feedback_signal (torch.Tensor or None): 反馈信号，形状可为任意，通常是风险评分等。
        """
        if feedback_signal is None or feedback_signal.numel() == 0:
            # 无反馈信号时不更新
            return

        # 计算反馈信号的均值，转为Python float
        feedback_mean = feedback_signal.mean().item()

        # 以0.5为基准，反馈越高，噪声强度增加，反馈越低，噪声强度减少
        adjustment = 0.05 * (feedback_mean - 0.5)  # 调整步长可调

        # 计算新的噪声强度
        new_noise_level = self.simple_noise_level + adjustment

        # 限制噪声强度在合理范围内，避免过大或过小
        new_noise_level = max(0.01, min(new_noise_level, 1.0))

        # 更新参数
        self.simple_noise_level = new_noise_level

        # 打印调试信息（可选）
        print(f"[SEU shm_update] feedback_mean: {feedback_mean:.4f}, "
              f"adjustment: {adjustment:.4f}, new_noise_level: {self.simple_noise_level:.4f}")

    #       def shm_update(self, feedback_signal=None): # Removed for simplification
    #            pass上面


class PrivacyDetectionUnit(nn.Module):
    def __init__(self, sensitive_keywords: list[str], device: str, feature_dim: int = 512,
                 bert_model_name='dmis-lab/biobert-base-cased-v1.1', gat_heads=4, gat_out_channels=128):
        super().__init__()
        self.sensitive_keywords = [kw.lower() for kw in sensitive_keywords]
        self.device = device
        self.feature_dim = feature_dim

        # BioBERT 文本编码器和 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.biobert = BertModel.from_pretrained(bert_model_name).to(device)

        # 线性层将BERT输出768维映射到feature_dim
        self.text_proj = nn.Linear(768, feature_dim)

        # GATConv 图编码器
        self.gat1 = GATConv(in_channels=feature_dim, out_channels=gat_out_channels, heads=gat_heads, concat=True)
        self.gat2 = GATConv(in_channels=gat_out_channels * gat_heads, out_channels=feature_dim, heads=1, concat=False)

        # 多头注意力融合文本和图特征
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, batch_first=True)

        # 分类器，输出风险评分
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

    def keyword_risk_score(self, text_prompts):
        """
        基于规则的关键词检测风险评分
        """
        batch_scores = []
        for prompt in text_prompts:
            prompt_lower = prompt.lower()
            hits = sum(1 for kw in self.sensitive_keywords if kw in prompt_lower)
            score = min(1.0, hits / max(1, len(self.sensitive_keywords)))  # 归一化
            batch_scores.append(score)
        return torch.tensor(batch_scores, device=self.device).unsqueeze(1)  # [B,1]

    def forward(self, text_prompts: list[str], concept_graph):
        batch_size = len(text_prompts)

        # 1. 文本编码
        encoding = self.tokenizer(text_prompts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        bert_outputs = self.biobert(**encoding)
        text_feats = bert_outputs.last_hidden_state[:, 0, :]  # [B, 768]
        text_feats = self.text_proj(text_feats)  # [B, feature_dim]

        # 2. 图编码
        x, edge_index, batch = concept_graph.x, concept_graph.edge_index, concept_graph.batch
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)  # [num_nodes, feature_dim]
        graph_feats = global_mean_pool(x, batch)  # [B, feature_dim]

        # 3. 融合机制
        query = text_feats.unsqueeze(1)  # [B, 1, feature_dim]
        key = graph_feats.unsqueeze(1)   # [B, 1, feature_dim]
        value = graph_feats.unsqueeze(1) # [B, 1, feature_dim]
        attn_output, _ = self.cross_attention(query, key, value)  # [B, 1, feature_dim]
        attn_output = attn_output.squeeze(1)  # [B, feature_dim]

        # 4. 分类器输出学习风险评分
        learned_risk_scores = self.classifier(attn_output)  # [B, 1]

        # 5. 计算基于规则的关键词风险评分
        rule_risk_scores = self.keyword_risk_score(text_prompts)  # [B, 1]

        # 6. 混合风险评估（加权平均）
        alpha = 0.5  # 权重，可调节
        final_risk_scores = alpha * learned_risk_scores + (1 - alpha) * rule_risk_scores

        return final_risk_scores, attn_output

    # adversarial_training 方法保持不变，模拟进化 (though may not be used with simplified PDU)


class ImmuneMemoryModule(nn.Module):
    """
    免疫记忆模块
    使用神经辐射场 (NeRF) 启发的傅里叶特征编码过去的隐私威胁事件/模式。
    模拟免疫系统的记忆能力。
    """
    def __init__(self, input_dim=512, fourier_dim=10, memory_dim=256, similarity_threshold=0.7):
        super().__init__()
        self.input_dim = input_dim # 输入特征维度 (来自 PDU 的 combined_features)
        self.fourier_dim = fourier_dim
        self.memory_dim = memory_dim # 存储在记忆库中的特征维度
        self.similarity_threshold = similarity_threshold

        # 傅里叶特征映射矩阵 B (式8)
        self.register_buffer('B', torch.randn(self.input_dim, self.fourier_dim) * 1.0) # 调整尺度

        # 记忆网络 (将傅里叶特征编码为记忆向量)
        fourier_output_dim = 2 * self.fourier_dim # sin 和 cos
        self.memory_net = nn.Sequential(
            nn.Linear(fourier_output_dim, 512), # 输入维度调整
            nn.ReLU(),
            nn.Linear(512, self.memory_dim) # 输出维度为 memory_dim
        )

        # 记忆库 (存储过去的威胁模式)
        self.memory_bank = None # 初始化为空

    def fourier_feature(self, x):
        """生成傅里叶特征 (式8)"""
        # x shape: [batch_size, input_dim]
        # B shape: [input_dim, fourier_dim]
        x_proj = x @ self.B # [batch_size, fourier_dim]
        # 移除 2*pi, 因为 B 是随机初始化的，效果类似
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # [batch_size, 2 * fourier_dim]

    def encode(self, features):
        """将输入特征编码为记忆向量"""
        # features shape: [batch_size, input_dim] (e.g., PDU's combined_features)
        fourier_features = self.fourier_feature(features) # [batch_size, 2 * fourier_dim]
        memory_vector = self.memory_net(fourier_features) # [batch_size, memory_dim]
        return memory_vector

    def update_memory(self, new_event_features):
        """
        更新免疫记忆库 (式9 启发)
        Args:
            new_event_features (torch.Tensor): 新检测到的高风险事件特征 [batch_size, input_dim] (来自 PDU)
        """
        if new_event_features is None or new_event_features.numel() == 0:
            return

        new_memory_vectors = self.encode(new_event_features) # [batch_size, memory_dim]

        if self.memory_bank is None or self.memory_bank.numel() == 0:
            self.memory_bank = new_memory_vectors.detach() # 存储时分离计算图
            return

        # 计算新事件与记忆库中向量的相似度
        sim_matrix = F.cosine_similarity(
            new_memory_vectors.unsqueeze(1), # [batch_size, 1, memory_dim]
            self.memory_bank.unsqueeze(0),    # [1, num_memory, memory_dim]
            dim=-1
        ) # [batch_size, num_memory]

        updated_memory = self.memory_bank.clone()
        merged_new_vectors = []
        used_new_indices = set()

        # 优先合并相似度高的
        for i in range(sim_matrix.size(0)): # 遍历新事件
            max_sim, max_idx = torch.max(sim_matrix[i], dim=0)
            if max_sim > self.similarity_threshold:
                # 合并：强化现有记忆 (类似亲和力成熟)
                updated_memory[max_idx] = F.normalize(0.8 * updated_memory[max_idx] + 0.2 * new_memory_vectors[i], dim=0)
                used_new_indices.add(i)

        # 将未合并的新事件添加到记忆库 (如果空间允许)
        for i in range(new_memory_vectors.size(0)):
            if i not in used_new_indices:
                 merged_new_vectors.append(new_memory_vectors[i].detach()) # 分离计算图

        if merged_new_vectors:
            updated_memory = torch.cat([updated_memory] + merged_new_vectors, dim=0)

        max_memory_size = 1024 # 示例大小
        if updated_memory.size(0) > max_memory_size:
             updated_memory = updated_memory[-max_memory_size:] # 保留最新的

        self.memory_bank = updated_memory


    def query_memory(self, query_features):
        """
        查询记忆库中与当前事件相似的威胁模式
        Args:
            query_features (torch.Tensor): 当前事件的特征 [batch_size, input_dim] (来自 PDU)
        Returns:
            torch.Tensor or None: 相似的记忆向量 (可用于指导 SEU)
        """
        if self.memory_bank is None or self.memory_bank.numel() == 0 or query_features is None:
            return None

        query_vectors = self.encode(query_features) # [batch_size, memory_dim]
        sim_matrix = F.cosine_similarity(
            query_vectors.unsqueeze(1), # [batch_size, 1, memory_dim]
            self.memory_bank.unsqueeze(0), # [1, num_memory, memory_dim]
            dim=-1
        ) # [batch_size, num_memory]

        # 这里简单返回每个查询最相似的记忆向量
        max_sim, max_indices = torch.max(sim_matrix, dim=1)
        relevant_memory = self.memory_bank[max_indices]

        # 可以根据相似度阈值过滤
        relevant_memory[max_sim < self.similarity_threshold] = 0 # 不相关的置零或移除

        return relevant_memory


# 移除了 NeRFEncoder 中的 self_destruct，因为这个机制（细胞凋亡）
# 更适合在主模型检测到高持续风险或对抗攻击时触发，可能涉及禁用某些层或连接。
# 可以创建一个独立的 ApoptosisModule 或在主训练循环中实现该逻辑。

# --- Placeholder Modules/Functions for Advanced Defense ---

class ApoptosisMechanism:
    """
    细胞凋亡机制 (Placeholder)
    """
    def __init__(self, risk_threshold=0.95, trigger_patience=5):
        self.risk_threshold = risk_threshold
        self.trigger_patience = trigger_patience
        self.high_risk_counter = 0

    def check_and_trigger(self, current_risk_score, model_unet):
        if current_risk_score > self.risk_threshold:
            self.high_risk_counter += 1
        else:
            self.high_risk_counter = 0

        if self.high_risk_counter >= self.trigger_patience:
            print("Apoptosis Triggered: High risk detected consistently.")
            # Placeholder: 实际应禁用 UNet 中的某些层或连接
            # e.g., for layer in model_unet.attention_layers: layer.enabled = False
            self.high_risk_counter = 0 # Reset after triggering
            return True # Indicates triggered
        return False

def epigenetic_prompt_encoding(prompt: str) -> str:
    """
    表观遗传启发的提示词加密 (Placeholder)
    """
    # 简单示例：字符替换或添加特殊标记
    encoded_prompt = prompt.replace("a", "@").replace("s", "$") + " [SECURE_TAG]"
    print(f"Epigenetic Encoding (Placeholder): '{prompt}' -> '{encoded_prompt}'")
    return encoded_prompt

# Quorum Quenching (群体淬灭) 本质上是架构层面的，涉及多个模型实例，
# 不适合在单个模型文件中实现。需要在分布式训练/推理框架中设计。

# --- Main ImmunoDiffusion Wrapper (Conceptual) ---
class ImmunoDiffusionModel(nn.Module):
    def __init__(self, pdu_config, seu_config, memory_config, apoptosis_config=None, model_id="runwayml/stable-diffusion-v1-5", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device

        # 加载基础扩散模型组件
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device)
        self.scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # 冻结 VAE 和 Text Encoder 的参数，因为我们通常不训练它们
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # UNet 的参数通常是可训练的，或者部分可训练 (例如，只训练 LoRA 层)

        # 初始化免疫模块
        # 确保 SEU 的 latent_dim 与 VAE 的潜空间维度一致
        # Stable Diffusion v1.5 VAE 的潜空间通道数为 4
        if 'latent_dim' not in seu_config:
             seu_config['latent_dim'] = self.vae.config.latent_channels
        self.pdu = PrivacyDetectionUnit(**pdu_config).to(self.device)
        self.seu = PrivacyEnhancementUnit(**seu_config).to(self.device)
        self.memory = ImmuneMemoryModule(**memory_config).to(self.device)
        self.apoptosis = ApoptosisMechanism(**apoptosis_config) if apoptosis_config else None

        # 获取UNet的输出通道数，用于调整SEU的输入 (如果SEU设计为操作UNet的输出而非潜变量本身)
        # 当前SEU设计为操作潜变量 z，所以 latent_dim 是关键

        # PDU的embed_dim 需要和text_encoder的输出兼容 (或者 PDU 内部的 text_proj 负责对齐)
        # CLIPTextModel (e.g., 'openai/clip-vit-large-patch14') typically outputs 768.
        # PDU's text_proj handles this: nn.Linear(768, self.embed_dim)
        # PDU's concept_encoder GATConv output is self.embed_dim // 2 * heads, which should be self.embed_dim
        # PDU's cross_attention embed_dim is self.embed_dim

    def _encode_prompt(self, prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0] # [batch_size, seq_len, embed_dim]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            uncond_tokens: list[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            
            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)
            
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def _prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            if generator is not None:
                latents = torch.randn(shape, generator=generator, device=self.device, dtype=dtype)
            else:
                latents = torch.randn(shape, device=self.device, dtype=dtype)
        else:
            latents = latents.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def vae_scale_factor(self):
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    def _decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def _numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images


    @torch.no_grad() # 通常推理时不需要梯度
    def forward(self, prompt: list[str] | str, concept_graph_data_list: list = None, # PDU 需要 concept_graph
                height: int = 512, width: int = 512, num_inference_steps: int = 50,
                guidance_scale: float = 7.5, negative_prompt: list[str] | str = None,
                num_images_per_prompt: int = 1, generator: torch.Generator | None = None,
                output_type: str = "pil", # "pil", "latent"
                pdu_text_input_override: dict = None # 允许直接传入PDU的文本输入，否则从prompt构造
               ):

        if isinstance(prompt, str):
            prompt = [prompt]
        if negative_prompt is not None and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        
        batch_size = len(prompt)

        # 1. 文本编码 (Classifier-Free Guidance)
        text_embeddings = self._encode_prompt(prompt, num_images_per_prompt, True, negative_prompt)

        # 2. PDU 处理：风险评估和特征提取
        # PDU 需要 tokenized input (e.g., BioBERT input) and concept_graph
        # 我们需要将主模型的 prompt (原始文本) 适配给 PDU 的 BioBERT
        # 假设 PDU 的 BioBERT 使用与主模型不同的 tokenizer 或需要特定格式
        if pdu_text_input_override:
            pdu_text_input = pdu_text_input_override
        else:
            # 简化：使用 PDU 自己的 tokenizer (如果它内部有的话) 或一个通用方式
            # 为了演示，这里假设 PDU 的 biobert 也能处理 tokenizer 输出的 input_ids 和 attention_mask
            # 但在实际中，PDU 的 biobert 可能有自己的 tokenizer
            # 这里我们暂时使用主 tokenizer 的结果，PDU 的实现应能处理它
            # 注意：PDU 的 forward 可能需要修改以接受与 CLIPTokenizer 不同的输入
            # 这里我们创建一个虚拟的 concept_graph，实际应用中需要用户提供
            # 此处我们仅为每个 batch item 准备文本输入
            pdu_input_ids_list = []
            pdu_attention_mask_list = []
            
            # PDU 内部的 BioBERT 需要自己的 tokenizer，这里为了代码能跑通，
            # 我们假设它和 CLIP tokenizer 输出的字典格式兼容，或者 PDU.forward 里处理。
            # 正确的做法是 PDU 应该暴露一个 `process_text` 方法或在 `__init__` 时接收 tokenizer
            temp_tokenizer_for_pdu = self.tokenizer # 临时代用
            for p_idx, p_item in enumerate(prompt):
                tokenized_pdu_text = temp_tokenizer_for_pdu(
                    p_item, padding="max_length", max_length=77, # BioBERT 通常有自己的 max_length
                    truncation=True, return_tensors="pt"
                )
                pdu_input_ids_list.append(tokenized_pdu_text.input_ids.squeeze(0))
                pdu_attention_mask_list.append(tokenized_pdu_text.attention_mask.squeeze(0))

            # 将列表堆叠成批次
            pdu_biobert_input = {
                'input_ids': torch.stack(pdu_input_ids_list).to(self.device),
                'attention_mask': torch.stack(pdu_attention_mask_list).to(self.device)
            }
        
        # 准备 concept_graph batch
        # 假设 concept_graph_data_list 是一个 Data 对象的列表，长度与 batch_size 相同
        # 如果为 None, PDU 需要能处理这种情况
        current_concept_graph = None
        if concept_graph_data_list:
            from torch_geometric.data import Batch
            try:
                current_concept_graph = Batch.from_data_list(concept_graph_data_list).to(self.device)
            except Exception as e:
                print(f"Warning: Could not create batch from concept_graph_data_list: {e}. PDU might not use graph features.")
                current_concept_graph = None # fallback
        
        # 如果 concept_graph_data_list 为空或处理失败，PDU 需要一个占位符或能优雅处理
        # PDU 的 forward 逻辑需要确保在 concept_graph 为 None 时不会崩溃
        # (其当前实现会尝试 .mean(dim=0) 如果 batch is None, 但 x 可能不存在)
        # 为了能运行，如果 current_concept_graph 是 None, 我们需要传递一个 PDU 能处理的结构
        if current_concept_graph is None:
            # 创建一个最小的、空的 Data 对象，PDU.forward 应该有鲁棒性处理
            from torch_geometric.data import Data
            # PDU 的 concept_encoder in_channels 是 concept_dim，默认为 200
            # PDU 的 embed_dim 默认为 512
            # GATConv in_channels=concept_dim, out_channels=embed_dim // 2
            # 我们需要确保 dummy_x 有正确的第二维度 (concept_dim)
            concept_dim_pdu = self.pdu.concept_encoder.in_channels 
            dummy_x = torch.empty((0, concept_dim_pdu), device=self.device) # 0个节点，但有正确的特征维度
            dummy_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            current_concept_graph = Data(x=dummy_x, edge_index=dummy_edge_index, batch=None).to(self.device)


        risk_score, combined_features = self.pdu(prompt, current_concept_graph)
        # risk_score: [batch_size * num_images_per_prompt, 1]
        # combined_features: [batch_size * num_images_per_prompt, pdu.embed_dim * 2]
        # 如果 PDU 的 batch size 与 text_embeddings 不一致 (因为 CFG)，需要调整
        # PDU 的输入是原始 prompt，输出 batch_size。text_embeddings 是 batch_size * 2 (CFG) * num_img
        # 我们假设PDU只针对原始prompt评估风险，然后将风险应用到所有图像和CFG上
        # 或者，PDU应该在_encode_prompt之后，对包括uncond的embeddings进行评估
        # 为简单起见，假设PDU的风险评分和特征是针对每个原始prompt的，需要扩展
        
        actual_batch_size_for_unet = text_embeddings.shape[0] # This is batch_size * num_images_per_prompt * (2 if CFG else 1)
        
        # 扩展 risk_score 和 combined_features 以匹配 UNet 的批次大小
        # PDU 的输出是 [batch_size, 1] 和 [batch_size, feature_dim]
        # 我们需要扩展到 [actual_batch_size_for_unet, ...]
        # 如果启用了CFG，UNet的输入批次是 (uncond + cond) * num_images_per_prompt
        # risk_score 和 combined_features 应该只基于 cond prompt 计算，然后广播
        
        # 假设 PDU 的输出 batch_size 已经等于 len(prompt) * num_images_per_prompt
        # 如果 PDU 只处理了原始 prompt (len(prompt))，需要扩展：
        if risk_score.shape[0] == batch_size:
            risk_score = risk_score.repeat_interleave(num_images_per_prompt, dim=0)
            combined_features = combined_features.repeat_interleave(num_images_per_prompt, dim=0)

        # 如果启用了CFG，还需要为uncond部分复制/生成risk_score和features
        # 简单处理：对uncond部分使用零风险/零特征，或复制cond的风险
        if guidance_scale > 1.0: # CFG enabled
            # 当前 risk_score 和 combined_features 是 [batch_size * num_images_per_prompt, ...]
            # 需要变成 [2 * batch_size * num_images_per_prompt, ...]
            # 对于 uncond 部分，可以假设风险为0或复制cond的风险。这里复制。
            risk_score_uncond = risk_score.clone() # Or torch.zeros_like(risk_score)
            risk_score = torch.cat([risk_score_uncond, risk_score])

            combined_features_uncond = combined_features.clone() # Or torch.zeros_like(combined_features)
            combined_features = torch.cat([combined_features_uncond, combined_features])


        # 3. 更新/查询免疫记忆模块
        # update_memory 应该只用高风险的条件提示特征
        # query_memory 可以用所有条件提示的特征
        # 假设 combined_features 现在是 [2 * B_eff, dim], 我们只需要 B_eff (条件部分)
        cond_combined_features = combined_features[batch_size * num_images_per_prompt:] if guidance_scale > 1.0 else combined_features
        # 假设 risk_score 也是 [2 * B_eff, 1], cond_risk_score 是后半部分
        cond_risk_score = risk_score[batch_size * num_images_per_prompt:] if guidance_scale > 1.0 else risk_score

        # 根据风险决定是否更新记忆 (例如，只用平均风险大于阈值的特征)
        # PDU 返回的 combined_features 是 [batch_size, embed_dim*2]
        # ImmuneMemoryModule 的 input_dim 应与此匹配
        if cond_risk_score.mean() > 0.5: # 示例阈值
             self.memory.update_memory(cond_combined_features)
        memory_signal = self.memory.query_memory(cond_combined_features) # [B_eff, memory_dim]

        if memory_signal is not None and guidance_scale > 1.0:
            # 为CFG的uncond部分添加空的memory_signal
            memory_signal_uncond = torch.zeros_like(memory_signal)
            memory_signal = torch.cat([memory_signal_uncond, memory_signal])
        elif memory_signal is None and actual_batch_size_for_unet > 0 : # 确保即使memory_signal是None，后续代码也能处理
             # memory_dim 来自 ImmuneMemoryModule
             mem_dim = self.memory.memory_dim
             memory_signal = torch.zeros(actual_batch_size_for_unet, mem_dim, device=self.device)


        # 4. 检查细胞凋亡
        if self.apoptosis:
            # 使用条件提示的平均风险
            avg_cond_risk = cond_risk_score.mean().item()
            if self.apoptosis.check_and_trigger(avg_cond_risk, self.unet):
                print("Generation halted due to Apoptosis.")
                if output_type == "pil":
                    # 返回一个表示错误的图像或空列表
                    # 创建一个简单的黑色图像作为占位符
                    error_img = Image.new('RGB', (width, height), color = 'black')
                    return [error_img] * (batch_size * num_images_per_prompt)
                elif output_type == "latent":
                    return torch.zeros(batch_size * num_images_per_prompt, self.vae.config.latent_channels, height // self.vae_scale_factor, width // self.vae_scale_factor, device=self.device) # 返回空潜变量

        # 5. 准备潜变量 (Initial Latents)
        num_channels_latents = self.unet.config.in_channels # 通常是 4 for SD
        latents = self._prepare_latents(
            batch_size * num_images_per_prompt, # 注意这里的batch_size是原始输入prompt数量
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype, # 通常是 float16 或 float32
            generator
        )

        # 6. 设置时间步
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # 7. 去噪循环
        for i, t in enumerate(timesteps):
            #扩展潜变量以匹配text_embeddings的批次大小 (CFG)
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 预测噪声 (UNet)
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

            # 执行 Classifier-Free Guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # SEU 应用：扰动潜变量
            # SEU 需要 risk_score 和 memory_signal
            # risk_score 和 memory_signal 需要与当前的 latents (去噪步骤输出) 的 batch_size 匹配
            # latents 是 [B_eff, C, H, W], risk_score 是 [2*B_eff, 1], memory_signal 是 [2*B_eff, mem_dim]
            # SEU应该只作用于条件部分，还是两者？如果两者，非条件部分风险为0？
            # 为简单起见，假设SEU接收与noise_pred相同batch size的risk_score和memory_signal
            # 但risk_score和memory_signal是基于原始prompt生成的，然后扩展到CFG
            # 此时的noise_pred是 [B_eff, C, H, W] (经过CFG合并后)
            # 我们需要 B_eff 大小的 risk_score (条件风险) 和 memory_signal (条件记忆)
            
            # 从CFG扩展的risk_score和memory_signal中取条件部分
            current_risk_for_seu = cond_risk_score # [B_eff, 1]
            current_memory_for_seu = memory_signal[batch_size * num_images_per_prompt:] if guidance_scale > 1.0 and memory_signal is not None else memory_signal
            if current_memory_for_seu is not None and current_memory_for_seu.shape[0] != latents.shape[0]: # B_eff
                 # 如果 memory_signal 在 CFG 情况下未正确处理或为None后生成了错误尺寸的zeros
                 if memory_signal is not None and guidance_scale > 1.0: #  memory_signal是 [2*B_eff, dim]
                     current_memory_for_seu = memory_signal[latents.shape[0]:]
                 elif memory_signal is None: # 如果一开始就是None
                     current_memory_for_seu = None # 保持None

            # 确保SEU的risk_score和latents的batch size一致
            if current_risk_for_seu.shape[0] != latents.shape[0]:
                 # Fallback: use mean risk or zero risk if dimensions mismatch
                 print(f"Warning: SEU risk_score batch mismatch. Latents: {latents.shape[0]}, Risk: {current_risk_for_seu.shape[0]}. Using mean risk.")
                 current_risk_for_seu = current_risk_for_seu.mean(dim=0, keepdim=True).repeat(latents.shape[0], 1)


            perturbed_noise_pred = self.seu(noise_pred, t, current_risk_for_seu, current_memory_for_seu)
            # SEU 的 forward 返回的是 z_perturbed，这里我们假设它扰动的是 noise_pred
            # 或者 SEU 应该在 scheduler.step 之后操作 latents
            # 按照原始论文的思路，SEU 是在扩散的某一步应用防御，操作的是 z (latent)
            # 所以 SEU 应该在 scheduler.step 之后。
            # 然而，更常见的做法是在噪声预测上加扰动，或者修改 UNet 行为。
            # 为了简单集成，这里假设 SEU 修改的是 noise_pred。如果SEU要修改latent，需要调整位置。
            # 如果 SEU.forward 返回的是 z_perturbed (潜变量)，那么它应该这样调用：
            # latents_before_seu = self.scheduler.step(noise_pred, t, latents).prev_sample
            # latents = self.seu(latents_before_seu, t, current_risk_for_seu, current_memory_for_seu)

            # 当前 SEU.forward(z, t, risk_score, memory_signal) 返回 z_perturbed
            # 为了使其适配当前循环，让 SEU 扰动的是 latents
            # 我们先用 scheduler.step 得到 prev_sample，然后应用 SEU

            # 计算上一步的潜变量 (Scheduler)
            prev_latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # 应用SEU到上一步的潜变量
            latents = self.seu(prev_latents, t, current_risk_for_seu, current_memory_for_seu)


        # 8. 解码图像 (VAE)
        if output_type == "latent":
            return latents

        images = self._decode_latents(latents)

        # 9. 转换为 PIL 图像
        if output_type == "pil":
            images = self._numpy_to_pil(images)

        return images


# class ImmunoDiffusionModel(nn.Module):
#     def __init__(self, base_diffusion_model, pdu_config, seu_config, memory_config, apoptosis_config):
#         super().__init__()
#         self.diffusion_model = base_diffusion_model # e.g., Stable Diffusion UNet+VAE+TextEncoder
#         self.pdu = PrivacyDetectionUnit(**pdu_config)
#         self.seu = PrivacyEnhancementUnit(**seu_config)
#         self.memory = ImmuneMemoryModule(**memory_config)
#         self.apoptosis = ApoptosisMechanism(**apoptosis_config) if apoptosis_config else None

#     def forward(self, text_prompt, concept_graph=None, timesteps=None, latents=None, ...):
#         # 1. (Optional) Encode prompt epigenetically
#         # encoded_prompt = epigenetic_prompt_encoding(text_prompt)

#         # 2. Process prompt with PDU (using diffusion model's text encoder or PDU's own)
#         # Assume text_input prepared for PDU's BioBERT
#         text_input = self.prepare_text_input(text_prompt) # This needs to be defined
#         risk_score, combined_features = self.pdu(text_input, concept_graph)

#         # 3. Update/Query Memory
#         # Only update memory for risky prompts?
#         if risk_score.mean() > 0.5: # Example threshold
#              self.memory.update_memory(combined_features)
#         memory_signal = self.memory.query_memory(combined_features)

#         # 4. Check for Apoptosis
#         if self.apoptosis and self.apoptosis.check_and_trigger(risk_score.mean(), self.diffusion_model.unet): # Pass unet part of diffusion_model
#              # Handle triggered state (e.g., return error, generate placeholder)
#              print("Generation halted due to Apoptosis.")
#              return None # Or a default safe image

#         # 5. Run Diffusion Process (Conceptual Loop)
#         # This needs full implementation using self.diffusion_model components (scheduler, unet)
#         # and integrating self.seu
#         # Example:
#         # latents = initial_latents
#         # self.scheduler.set_timesteps(num_inference_steps)
#         # for t in self.scheduler.timesteps:
#         #     # Prepare input for UNet (handle CFG if used)
#         #     unet_input = torch.cat([latents] * 2) if do_cfg else latents
#         #     unet_input = self.scheduler.scale_model_input(unet_input, t)
#         #
#         #     # Predict noise
#         #     noise_pred = self.diffusion_model.unet(unet_input, t, encoder_hidden_states=text_embeddings).sample
#         #
#         #     # CFG
#         #     if do_cfg:
#         #        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#         #        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#         #
#         #     # Get previous sample
#         #     prev_sample = self.scheduler.step(noise_pred, t, latents).prev_sample
#         #
#         #     # Apply SEU
#         #     # Ensure risk_score and memory_signal are correctly shaped and aligned with prev_sample
#         #     latents = self.seu(prev_sample, t, risk_score, memory_signal) # SEU operates on latents
#         #
#         # # 6. Decode final latent with VAE
#         # image = self.diffusion_model.vae.decode(latents / self.diffusion_model.vae.config.scaling_factor).sample
#         # return image

        pass # This forward method needs full implementation within a diffusion framework
