import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
import torch.nn.init as init
import numpy as np


# class CrossAttention(nn.Module):
#     def __init__(self, feature_dim, dropout_prob=0.1):
#         super(CrossAttention, self).__init__()
#         self.text_linear = nn.Linear(feature_dim, feature_dim)
#         self.extra_linear = nn.Linear(feature_dim, feature_dim)
#         self.query_proj = nn.Linear(feature_dim, feature_dim)
#         self.key_proj = nn.Linear(feature_dim, feature_dim)
#         self.value_proj = nn.Linear(feature_dim, feature_dim)
#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(self, query, key, value):
#         if query.shape[-1] != 768:
#             query = self.text_linear(query)  # [batch, seq_len, feature_dim]
#         if key.shape[-1] != 768:
#             key = self.extra_linear(key)
#             value = self.extra_linear(value)
#         query = self.query_proj(query)  # [batch, feature_dim]
#         key = self.key_proj(key)  # [batch, feature_dim]
#         value = self.value_proj(value)  # [batch, feature_dim]
#         attention_scores = torch.matmul(query, key.transpose(-1, -2))
#         attention_scores = attention_scores / torch.sqrt(
#             torch.tensor(key.size(-1), dtype=torch.float32)
#         )
#         attention_weights = F.softmax(attention_scores, dim=-1)
#         attended_values = torch.matmul(attention_weights, value)
#         attended_values = self.dropout(attended_values)
#         return attended_values

#SDA block
class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):

        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


def build_activation_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    elif act_type == 'GELU':
        return nn.GELU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    elif act_type == 'SiLU':
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


class ElementWiseScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, channels, init_value=1e-5, requires_grad=True):
        super(ElementWiseScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, channels, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

#FDA block
class SFDABlock(nn.Module):
    """
    An improved module to reduce redundancy and enhance uniqueness in image features.
    """

    # Scaled Feature Decomposition and Aggregation Module
    def __init__(self,
                 embed_dims,
                 reduction_ratio=16,
                 act_type='GELU',
                 dropout_rate=0.0):
        super(SFDABlock, self).__init__()

        self.embed_dims = embed_dims
        self.intermediate_channels = int(embed_dims // 2)

        # Initial convolution layers to process input features
        self.conv_initial = nn.Sequential(
            nn.Conv2d(in_channels=embed_dims,
                      out_channels=self.intermediate_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(self.intermediate_channels),
            build_activation_layer(act_type)
        )

        # Attention mechanism to focus on unique features
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.intermediate_channels, self.intermediate_channels // reduction_ratio, 1, bias=False),
            build_activation_layer(act_type),
            nn.Conv2d(self.intermediate_channels // reduction_ratio, self.intermediate_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.intermediate_channels, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )

        # Feature decomposition to reduce redundancy
        self.decompose_conv = nn.Conv2d(
            in_channels=self.intermediate_channels,
            out_channels=1,
            kernel_size=1
        )
        self.sigma = ElementWiseScale(self.intermediate_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_activation_layer(act_type)

        # Final projection to original dimensions
        self.conv_final = nn.Sequential(
            nn.Conv2d(in_channels=self.intermediate_channels,
                      out_channels=embed_dims,
                      kernel_size=1),
            nn.BatchNorm2d(embed_dims),
            build_activation_layer(act_type)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def feat_decompose(self, x):
        """
        Feature decomposition to reduce redundancy and enhance uniqueness.
        """
        x_global = self.decompose_act(self.decompose_conv(x))  # [B, 1, H, W]
        x = x + self.sigma(x - x_global)
        return x

    def forward(self, x):
        # Initial convolution
        x = self.conv_initial(x)

        # Apply channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Apply spatial attention
        sa = self.spatial_attention(x)
        x = x * sa

        # Feature decomposition
        x = self.feat_decompose(x)

        x = self.dropout(x)

        # Final projection
        x = self.conv_final(x)
        x = self.dropout(x)

        return x

#DMI block
class DynamicAdaptiveInteractionBlock(nn.Module):
    def __init__(self, feature_dim, dropout_prob=0.1):
        super(DynamicAdaptiveInteractionBlock, self).__init__()
        self.text_linear = nn.Linear(feature_dim, feature_dim)
        self.extra_linear = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.text_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        self.image_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        # 自适应的缩放因子
        self.text_scale = nn.Parameter(torch.ones(1, feature_dim), requires_grad=True)
        self.image_scale = nn.Parameter(torch.ones(1, feature_dim), requires_grad=True)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, mode='text'):
        # 确保输入的特征维度为 768
        if query.shape[-1] != 768:
            query = self.text_linear(query)
        if key.shape[-1] != 768:
            key = self.extra_linear(key)
            value = self.extra_linear(value)

        # 对 query, key, value 进行线性投影
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # 计算注意力得分并应用 softmax
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算 attended_values
        attended_values = torch.matmul(attention_weights, value)
        attended_values = self.dropout(attended_values)

        # 根据 mode 使用不同的门控和自适应缩放因子
        if mode == 'text':
            gated_values = torch.sigmoid(self.text_gate(attended_values)) * attended_values * self.text_scale
        elif mode == 'image':
            gated_values = torch.sigmoid(self.image_gate(attended_values)) * attended_values * self.image_scale
        else:
            raise ValueError("Invalid mode. Choose either 'text' or 'image'.")

        return gated_values


class MV_CLIP_new(nn.Module):
    def __init__(self, args):
        super(MV_CLIP_new, self).__init__()

        # 加载 CLIP
        self.model = CLIPModel.from_pretrained("/home/wyj/.cache/huggingface/hub/models--openai--clip-vit-base-patch32",
                                               local_files_only=True)

        # 定义文本和图像特征的线性映射层
        self.text_linear = nn.Sequential(
            nn.Linear(args.text_size, 768),
            nn.Dropout(args.dropout_rate),
            nn.GELU()
        )
        self.image_linear = nn.Sequential(
            nn.Linear(args.image_size, args.image_size),
            nn.Dropout(args.dropout_rate),
            nn.GELU()
        )

        # 使用 ScaledDotProductAttention 处理文本特征
        self.self_attention_layer_text = ScaledDotProductAttention(d_model=768, d_k=64, d_v=64, h=8, dropout=0.1)

        # 使用 FDA(SFDA) 聚合图像特征
        self.channel_ffn = SFDABlock(embed_dims=args.image_size)

        # 使用 DMI 替代原来的 CrossAttention
        self.cross_att = DynamicAdaptiveInteractionBlock(feature_dim=768, dropout_prob=0.1)
        # 融合后的分类器
        self.classifier_fuse = nn.Linear(args.image_size, args.label_number)
        self.loss_fct = nn.CrossEntropyLoss()
        # 定义初始正则化权重
        self.initial_lambda = 0.5

    def jensen_shannon_divergence(self, p_logits, q_logits):
        # Convert logits to probabilities
        p_probs = F.softmax(p_logits, dim=-1)
        q_probs = F.softmax(q_logits, dim=-1)

        # Compute M distribution
        m_probs = 0.5 * (p_probs + q_probs)

        # Compute KL(P || M) and KL(Q || M)
        kl_pm = F.kl_div(F.log_softmax(p_logits, dim=-1), m_probs, reduction='batchmean')
        kl_qm = F.kl_div(F.log_softmax(q_logits, dim=-1), m_probs, reduction='batchmean')

        # Jensen-Shannon Divergence
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd

    def forward(self, inputs, labels=None):
        # 提取输入的文本和图像特征
        output = self.model(**inputs)
        text_feature = output['text_model_output']['pooler_output']
        image_feature = output['vision_model_output']['pooler_output']

        # 对文本和图像特征进行线性映射
        text_feature = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)
        # 使用 ScaledDotProductAttention 处理文本特征
        text_feature = self.self_attention_layer_text(text_feature.unsqueeze(1), text_feature.unsqueeze(1),
                                                      text_feature.unsqueeze(1)).squeeze(1)
        # 使用 FDA 对图像特征进行聚合 1
        image_feature = image_feature.unsqueeze(-1).unsqueeze(-1)  # (B, C) -> (B, C, 1, 1)
        image_feature = self.channel_ffn(image_feature).squeeze(-1).squeeze(-1)

        # Gated交叉注意力处理
        cross_feature_text = self.cross_att(text_feature, image_feature, image_feature, mode='text')
        cross_feature_image = self.cross_att(image_feature, text_feature, text_feature, mode='image')
        cross_feature_text = cross_feature_text.squeeze(1)  # [batch_size, 768]
        cross_feature_image = cross_feature_image.squeeze(1)  # [batch_size, 768]

        # fuse_feature = cross_feature_text#消融图像
        fuse_feature = cross_feature_text + cross_feature_image

        # 分类器预测
        logits_fuse = self.classifier_fuse(fuse_feature)
        fuse_score = F.softmax(logits_fuse, dim=-1)

        outputs = (fuse_score,)
        if labels is not None:
            # 分类损失
            loss_main = self.loss_fct(logits_fuse, labels)

            lambda_co_reg = self.initial_lambda * (1 / (1 + torch.exp(-loss_main)))
            js_divergence = self.jensen_shannon_divergence(text_feature, image_feature)
            # 合并总损失
            total_loss = loss_main + lambda_co_reg * js_divergence
            # total_loss = loss_main
            outputs = (total_loss,) + outputs

        return outputs
