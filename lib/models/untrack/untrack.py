#new_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import math

# 全局配置
DEFAULT_DIM = 768
DEFAULT_NUM_BANDS = 8
DEFAULT_MAX_MEMORY = 150
DEFAULT_PATCH_SIZE = 16
DEFAULT_SEARCH_SIZE = 384


#辅助模块

class BandAttention(nn.Module):
    """波段注意力：为每个波段分配动态权重"""

    def __init__(self, num_bands=DEFAULT_NUM_BANDS, dim=DEFAULT_DIM, ratio=4):
        super().__init__()
        self.num_bands = num_bands
        self.dim = dim

        # 修复：输入是dim（每个波段的特征维度），输出是1（权重）
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(dim // ratio, 1),  # 输出单个权重值
            nn.Softmax(dim=1)  # 在波段维度上归一化
        )

    def forward(self, x):
        """
        Args:
            x: [B, num_bands, C] 每个波段的特征表示
        Returns:
            weights: [B, num_bands, 1] 波段权重
        """
        # x: [B, num_bands, C]
        weights = self.mlp(x)  # [B, num_bands, 1]
        return weights


class ResidualSE(nn.Module):
    """残差SE门控"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 或 [B, H, W, C]
        Returns:
            out: 与输入同形状
        """
        # 统一处理为 [B, C, H, W]
        if x.dim() == 4 and x.shape[-1] == self.dim:
            # [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()
            need_permute_back = True
        else:
            need_permute_back = False

        # SE操作
        b, c, h, w = x.shape
        y = self.avg_pool(x).view(b, c)  # [B, C]
        y = self.fc(y).view(b, c, 1, 1)  # [B, C, 1, 1]
        out = x * y.expand_as(x)  # [B, C, H, W]

        # 残差连接
        out = out + x

        if need_permute_back:
            out = out.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        return out


#核心模块1：多光谱-多尺度级联融合

class MS_HFC(nn.Module):
    """多光谱-多尺度级联融合模块"""

    def __init__(self, dim=DEFAULT_DIM, num_bands=DEFAULT_NUM_BANDS):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands

        # 波段注意力
        self.band_attn = BandAttention(num_bands, dim)

        # 特征投影：将多波段特征投影到单特征
        self.band_proj = nn.Conv2d(dim * num_bands, dim, kernel_size=1, bias=False)

        # SE门控（用于级联后的特征）
        self.se_gate1 = ResidualSE(dim * 2)
        self.se_gate2 = ResidualSE(dim * 2)

        # 特征变换
        self.feat_transform = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, multi_scale_feats, band_feats):
        """
        Args:
            multi_scale_feats: list of [B, C, H, W] - 3个尺度的特征（从大到小）
            band_feats: [B, num_bands, C, H, W] - 多波段原始特征
        Returns:
            fused_feat: [B, C, H, W] - 融合后的特征（最大尺度）
        """
        assert len(multi_scale_feats) == 3, "需要3个尺度的特征"
        f1, f2, f3 = multi_scale_feats  # f1: 大尺度, f3: 小尺度
        B, C, H1, W1 = f1.shape
        _, _, H2, W2 = f2.shape
        _, _, H3, W3 = f3.shape

        # 1. 处理多波段特征
        # band_feats: [B, num_bands, C, H, W]
        # 为每个尺度生成波段加权特征

        def process_band_feat(target_h, target_w):
            # 插值到目标尺寸
            bf = F.interpolate(
                band_feats.flatten(0, 1),  # [B*num_bands, C, H, W]
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )  # [B*num_bands, C, H, W]

            # 恢复batch维度
            bf = bf.view(B, self.num_bands, C, target_h, target_w)

            # 全局池化获取每个波段的特征表示 [B, num_bands, C]
            band_global = F.adaptive_avg_pool2d(
                bf.flatten(0, 1), 1
            ).view(B, self.num_bands, C)

            # 获取波段权重
            band_weights = self.band_attn(band_global)  # [B, num_bands, 1]

            # 加权融合
            bf = bf * band_weights.view(B, self.num_bands, 1, 1, 1)  # [B, num_bands, C, H, W]
            bf = bf.sum(dim=1)  # [B, C, H, W]

            return bf

        # 为每个尺度生成波段特征
        f1_band = process_band_feat(H1, W1)  # [B, C, H1, W1]
        f2_band = process_band_feat(H2, W2)  # [B, C, H2, W2]
        f3_band = process_band_feat(H3, W3)  # [B, C, H3, W3]

        # 2. 多尺度级联融合（自顶向下）
        # 高尺度(f3) -> 中尺度(f2)
        f3_up = F.interpolate(f3_band, size=(H2, W2), mode='bilinear', align_corners=False)  # [B, C, H2, W2]

        # 修复：确保f2_band也是[B, C, H2, W2]
        if f2_band.shape[2:] != (H2, W2):
            f2_band = F.interpolate(f2_band, size=(H2, W2), mode='bilinear', align_corners=False)

        f2_fused = torch.cat([f2_band, f3_up], dim=1)  # [B, 2C, H2, W2]
        f2_fused = self.se_gate1(f2_fused)  # [B, 2C, H2, W2]
        f2_fused = f2_fused[:, :C] + f2_fused[:, C:]  # 简化：直接相加 [B, C, H2, W2]

        # 中尺度 -> 低尺度（大尺度）
        f2_up = F.interpolate(f2_fused, size=(H1, W1), mode='bilinear', align_corners=False)  # [B, C, H1, W1]

        if f1_band.shape[2:] != (H1, W1):
            f1_band = F.interpolate(f1_band, size=(H1, W1), mode='bilinear', align_corners=False)

        f1_fused = torch.cat([f1_band, f2_up], dim=1)  # [B, 2C, H1, W1]
        f1_fused = self.se_gate2(f1_fused)  # [B, 2C, H1, W1]
        f1_fused = f1_fused[:, :C] + f1_fused[:, C:]  # [B, C, H1, W1]

        # 特征变换
        out = self.feat_transform(f1_fused)  # [B, C, H1, W1]

        return out


# 核心模块2：长期光谱记忆库

class LSMB(nn.Module):
    """长期光谱记忆库"""

    def __init__(self, dim=DEFAULT_DIM, comp_dim=64, max_size=DEFAULT_MAX_MEMORY):
        super().__init__()
        self.dim = dim
        self.comp_dim = comp_dim
        self.max_size = max_size

        # 压缩卷积
        self.comp_conv = nn.Conv2d(dim, comp_dim, kernel_size=1, bias=False)

        # 记忆库（使用buffer以便保存）
        self.register_buffer("memory_keys", torch.zeros(0, comp_dim))
        self.register_buffer("memory_feats", torch.zeros(0, dim, 24, 24))
        self.register_buffer("memory_boxes", torch.zeros(0, 4))
        self.register_buffer("memory_confs", torch.zeros(0))

        # 可学习融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.3))

        # 相似度计算
        self.temp = nn.Parameter(torch.tensor(10.0))  # 温度系数

    def reset_memory(self):
        """重置记忆库"""
        device = self.memory_keys.device
        self.memory_keys = torch.zeros(0, self.comp_dim, device=device)
        self.memory_feats = torch.zeros(0, self.dim, 24, 24, device=device)
        self.memory_boxes = torch.zeros(0, 4, device=device)
        self.memory_confs = torch.zeros(0, device=device)

    def _update_memory(self, key, feat, box, conf):
        """更新记忆库（FIFO）"""
        # 检查是否已存在相似记忆
        if self.memory_keys.shape[0] > 0:
            sim = F.cosine_similarity(key, self.memory_keys, dim=-1)
            if sim.max() > 0.95:  # 过于相似，不添加
                return

        # FIFO
        if self.memory_keys.shape[0] >= self.max_size:
            self.memory_keys = self.memory_keys[1:]
            self.memory_feats = self.memory_feats[1:]
            self.memory_boxes = self.memory_boxes[1:]
            self.memory_confs = self.memory_confs[1:]

        # 添加新记忆
        self.memory_keys = torch.cat([self.memory_keys, key], dim=0)
        self.memory_feats = torch.cat([self.memory_feats, feat], dim=0)
        self.memory_boxes = torch.cat([self.memory_boxes, box], dim=0)
        self.memory_confs = torch.cat([self.memory_confs, conf], dim=0)

    def forward(self, curr_feat, curr_box, curr_conf):
        """
        Args:
            curr_feat: [B, C, H, W]
            curr_box: [B, 4] (cx, cy, w, h) 归一化坐标
            curr_conf: [B, 1] 或 [B]
        Returns:
            fused_feat: [B, C, H, W]
        """
        B, C, H, W = curr_feat.shape
        device = curr_feat.device

        # 确保记忆库在正确设备
        if self.memory_keys.device != device:
            self.memory_keys = self.memory_keys.to(device)
            self.memory_feats = self.memory_feats.to(device)
            self.memory_boxes = self.memory_boxes.to(device)
            self.memory_confs = self.memory_confs.to(device)

        # 压缩当前特征
        comp_curr = self.comp_conv(curr_feat)  # [B, comp_dim, H, W]
        comp_curr = F.adaptive_avg_pool2d(comp_curr, 1).view(B, self.comp_dim)  # [B, comp_dim]

        # 更新记忆库（逐样本）
        curr_conf_flat = curr_conf.view(B) if curr_conf.dim() > 1 else curr_conf
        for b in range(B):
            if curr_conf_flat[b] > 0.7:  # 置信度阈值
                # 调整特征尺寸
                feat_resized = F.interpolate(
                    curr_feat[b:b + 1],
                    size=(24, 24),
                    mode='bilinear',
                    align_corners=False
                )  # [1, C, 24, 24]

                self._update_memory(
                    comp_curr[b:b + 1],  # [1, comp_dim]
                    feat_resized,  # [1, C, 24, 24]
                    curr_box[b:b + 1],  # [1, 4]
                    curr_conf_flat[b:b + 1]  # [1]
                )

        # 如果记忆不足，直接返回当前特征
        if self.memory_keys.shape[0] < 3:
            return curr_feat

        # 计算相似度
        # 光谱相似度（余弦）
        spectral_sim = F.cosine_similarity(
            comp_curr.unsqueeze(1),  # [B, 1, comp_dim]
            self.memory_keys.unsqueeze(0),  # [1, N, comp_dim]
            dim=-1
        )  # [B, N]

        # 空间相似度（框的距离）
        # 转换为中心点
        curr_center = curr_box[:, :2]  # [B, 2]
        mem_center = self.memory_boxes[:, :2]  # [N, 2]
        spatial_dist = torch.cdist(curr_center, mem_center)  # [B, N]
        spatial_sim = torch.exp(-spatial_dist * 5)  # 高斯核 [B, N]

        # 综合相似度
        total_sim = 0.6 * spectral_sim + 0.4 * spatial_sim  # [B, N]

        # Top-K选择
        K = min(5, self.memory_keys.shape[0])
        topk_sim, topk_idx = torch.topk(total_sim, k=K, dim=-1)  # [B, K]

        # 加权聚合
        weights = F.softmax(topk_sim / self.temp, dim=-1)  # [B, K]

        # 聚合历史特征
        fused_hist = torch.zeros_like(curr_feat)  # [B, C, H, W]
        for b in range(B):
            hist_feats = self.memory_feats[topk_idx[b]]  # [K, C, 24, 24]
            w = weights[b].view(K, 1, 1, 1)  # [K, 1, 1, 1]
            hist_feat = (hist_feats * w).sum(dim=0)  # [C, 24, 24]

            # 插值到当前尺寸
            hist_feat = F.interpolate(
                hist_feat.unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [C, H, W]

            fused_hist[b] = hist_feat

        # 可学习融合
        w = torch.sigmoid(self.fusion_weight)
        out = (1 - w) * curr_feat + w * fused_hist

        return out


#核心模块3：动态自适应背景消除

def dynamic_background_elimination(attn_scores, keep_ratio=0.7,
                                   prev_boxes=None, curr_conf=None,
                                   feat_size=24, patch_size=16):
    """
    动态背景消除（改进版）
    Args:
        attn_scores: [B, N, C] 注意力分数（N=HW）
        keep_ratio: 基础保留比例
        prev_boxes: [B, 4] 上一帧预测框（归一化cx,cy,w,h）
        curr_conf: [B] 当前置信度
        feat_size: 特征图尺寸（默认24x24）
    Returns:
        kept_indices: list of [K] 保留的索引
        keep_ratio_actual: 实际保留比例
    """
    B, N, C = attn_scores.shape
    device = attn_scores.device

    # 计算每个token的重要性分数
    importance = attn_scores.norm(dim=-1)  # [B, N]

    # 动态调整保留比例
    if curr_conf is not None:
        # 高置信度时保留更多（更信任注意力）
        conf_factor = 0.8 + 0.4 * curr_conf.clamp(0, 1)  # [B]
        keep_ratio = keep_ratio * conf_factor

    keep_ratio = keep_ratio.clamp(0.3, 0.9)
    lens_keep = (keep_ratio * N).long().clamp(min=1, max=N)

    kept_indices = []
    for b in range(B):
        idx = torch.arange(N, device=device)

        # 如果提供了前一帧框，保护目标区域
        if prev_boxes is not None:
            box = prev_boxes[b]  # [4] cx,cy,w,h
            cx, cy, w, h = box

            # 转换为特征图坐标
            cx, cy = cx * feat_size, cy * feat_size
            w, h = w * feat_size, h * feat_size

            # 计算每个token的中心位置
            token_x = (idx % feat_size).float()
            token_y = (idx // feat_size).float()

            # 判断是否在目标区域内（扩展1.2倍）
            in_box = ((token_x >= cx - w * 0.6) & (token_x <= cx + w * 0.6) &
                      (token_y >= cy - h * 0.6) & (token_y <= cy + h * 0.6))

            # 优先保留框内token
            in_box_idx = idx[in_box]
            out_box_idx = idx[~in_box]

            need_out = max(0, lens_keep[b].item() - len(in_box_idx))

            if need_out == 0:
                kept_idx = in_box_idx
            else:
                # 从框外选高分token
                out_scores = importance[b, out_box_idx]
                _, top_out = torch.topk(out_scores, min(need_out, len(out_box_idx)))
                kept_idx = torch.cat([in_box_idx, out_box_idx[top_out]])
        else:
            # 直接取Top-K
            _, kept_idx = torch.topk(importance[b], lens_keep[b].item())

        kept_indices.append(kept_idx)

    return kept_indices, keep_ratio.mean().item()


# 改进版UNTrack（兼容原接口）

class UNTrackPro(nn.Module):
    """
    UNTrack改进版 - 保持与原UNTrack接口兼容
    新增功能：
    1. MS-HFC: 多光谱-多尺度级联融合
    2. LSMB: 长期光谱记忆库
    3. 动态背景消除
    """

    def __init__(self, backbone, box_head, aux_loss=False,
                 head_type="CENTER", token_len=1, num_searches=2,
                 use_mshfc=True, use_lsmb=True, use_dynamic_ce=True):
        super().__init__()

        # 原始组件
        self.backbone = backbone
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type
        self.token_len = token_len
        self.num_searches = num_searches

        if head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        # 历史提示
        self.track_query = None

        # 原始光谱提示编码器
        from timm.models.layers import Mlp
        self.prompt = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            Mlp(768, 768 * 4, 768)
        )

        # 新增模块开关
        self.use_mshfc = use_mshfc
        self.use_lsmb = use_lsmb
        self.use_dynamic_ce = use_dynamic_ce

        # 新增模块
        if use_mshfc:
            self.ms_hfc = MS_HFC(768, DEFAULT_NUM_BANDS)
            # 特征融合层
            self.feat_fusion = nn.Sequential(
                nn.Conv2d(768 * 2, 768, kernel_size=1),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )

        if use_lsmb:
            self.stm = LSMB(768, comp_dim=64, max_size=DEFAULT_MAX_MEMORY)

        # 多尺度特征提取（从ViT不同层）
        self.multi_scale_extractors = nn.ModuleList([
            nn.Identity(),  # 第3层
            nn.Identity(),  # 第6层
            nn.Identity(),  # 第9层
        ])

        # 记录上一帧结果用于动态CE
        self.prev_boxes = None
        self.prev_confs = None

    def reset_memory(self):
        """重置记忆库（新序列开始时调用）"""
        # if hasattr(self, 'lsmb'):
        #     self.lsmb.reset_memory()
        if hasattr(self, 'stm'):  # <--- 改为 self.stm
            self.stm.reset_memory()  # <--- 改为 self.stm
        self.track_query = None
        self.prev_boxes = None
        self.prev_confs = None

    def extract_multi_scale_feats(self, x, layer_indices=[2, 5, 8]):
        """
        从ViT中间层提取多尺度特征
        注意：需要修改backbone以返回中间层特征
        """
        # 简化版本：直接对最终特征进行不同尺度的池化
        feats = []
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x_2d = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

        # 大尺度（原始）
        feats.append(x_2d)  # [B, C, 24, 24]
        # 中尺度
        feats.append(F.avg_pool2d(x_2d, 2))  # [B, C, 12, 12]
        # 小尺度
        feats.append(F.avg_pool2d(x_2d, 4))  # [B, C, 6, 6]

        return feats

    def forward(self, template, search, ce_template_mask=None,
                ce_keep_rate=None, return_last_attn=False):
        """
        与原始UNTrack兼容的前向接口

        Args:
            template: list of [B, 8, 192, 192]
            search: list of [B, 8, 384, 384]（N帧）
        Returns:
            out_dict: list of dict，每帧的结果
        """
        out_dict = []

        for i in range(self.num_searches - 1, len(search)):
            # 获取搜索窗口
            search_window = [search[idx] for idx in range(i - self.num_searches + 1, i + 1)]

            # 动态调整ce_keep_rate
            actual_keep_rate = ce_keep_rate
            if self.use_dynamic_ce and self.prev_confs is not None:
                # 根据历史置信度动态调整
                avg_conf = self.prev_confs.mean()
                actual_keep_rate = ce_keep_rate * (0.8 + 0.4 * avg_conf)
                actual_keep_rate = max(0.5, min(0.9, actual_keep_rate))

            # 骨干网络前向
            x_, aux_dict, _ = self.backbone(
                z=template.copy(),
                x=search_window,
                ce_template_mask=ce_template_mask,
                ce_keep_rate=actual_keep_rate,
                return_last_attn=return_last_attn,
                track_query=self.track_query,
                token_len=self.token_len
            )

            # 提取搜索区域特征
            # x_: [B, token_len + template_len + search_len*N, C]
            feat_len_total = self.num_searches * self.feat_len_s
            search_start = self.token_len + (x_.shape[1] - self.token_len - feat_len_total)

            # 只取最后一个搜索帧的特征
            search_feat = x_[:, -self.feat_len_s:, :]  # [B, HW, C]
            template_feat = x_[:, self.token_len:search_start, :]  # [B, T*HW, C]
            prompt_feat = x_[:, :self.token_len, :]  # [B, token_len, C]

            # ==================== 新增：MS-HFC ====================
            if self.use_mshfc:
                # 提取多尺度特征
                multi_scale_feats = self.extract_multi_scale_feats(search_feat)

                # 从原始图像获取多波段特征（简化：使用search最后一帧）
                band_feat = search_window[-1]  # [B, 8, 384, 384]
                # 下采样到特征尺寸
                band_feat = F.interpolate(
                    band_feat.flatten(0, 1).unsqueeze(1),  # [B*8, 1, 384, 384]
                    size=(self.feat_sz_s, self.feat_sz_s),  # 24x24
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1).view(-1, 8, self.feat_sz_s, self.feat_sz_s)  # [B, 8, 24, 24]

                # 投影到特征维度
                band_feat = band_feat.unsqueeze(2).expand(-1, -1, 768, -1, -1)  # [B, 8, 768, 24, 24]

                # MS-HFC融合
                mshfc_out = self.ms_hfc(multi_scale_feats, band_feat)  # [B, 768, 24, 24]

                # 与原始特征融合
                search_feat_2d = search_feat.transpose(1, 2).view(-1, 768, self.feat_sz_s, self.feat_sz_s)
                fused_feat = torch.cat([search_feat_2d, mshfc_out], dim=1)
                fused_feat = self.feat_fusion(fused_feat)  # [B, 768, 24, 24]
                search_feat = fused_feat.flatten(2).transpose(1, 2)  # [B, HW, 768]

            # ==================== 更新光谱提示 ====================
            if self.backbone.add_cls_token:
                # 使用原始prompt编码器
                t_query = prompt_feat  # [B, 1, 768]
                z_query = template_feat  # [B, T*HW, 768]

                # 池化模板特征
                z_pooled = z_query.mean(dim=1, keepdim=True)  # [B, 1, 768]

                # 拼接并编码
                prompt_input = torch.cat([t_query, z_pooled], dim=-1)  # [B, 1, 1536]
                self.track_query = self.prompt(prompt_input).clone().detach()  # [B, 1, 768]

            # ==================== 注意力加权 ====================
            # 计算提示与搜索的注意力
            att = torch.matmul(search_feat, prompt_feat.transpose(1, 2))  # [B, HW, 1]
            opt = (search_feat.unsqueeze(-1) * att.unsqueeze(-2))  # [B, HW, C, 1]
            opt = opt.permute((0, 3, 2, 1)).contiguous()  # [B, 1, C, HW]

            # ==================== 预测头 ====================
            out = self.forward_head(opt, None)

            # ==================== 新增：LSMB ====================
            if self.use_lsmb:
                # 获取预测框和置信度
                pred_boxes = out['pred_boxes']  # [B, 1, 4]
                score_map = out['score_map']  # [B, 1, H, W]

                # 计算置信度
                conf = score_map.max(dim=-1)[0].max(dim=-1)[0]  # [B, 1]

                # 准备特征 [B, C, H, W]
                curr_feat = search_feat.transpose(1, 2).view(-1, 768, self.feat_sz_s, self.feat_sz_s)

                # LSMB融合
                fused_feat = self.stm(curr_feat, pred_boxes.squeeze(1), conf)

                # 重新预测（可选：使用融合后的特征再次预测）
                # 简化：仅更新记忆，不改变当前输出

                # 保存用于下一帧
                self.prev_boxes = pred_boxes.squeeze(1).detach()
                self.prev_confs = conf.squeeze(-1).detach()

            # 合并输出
            out.update(aux_dict)
            out['backbone_feat'] = x_
            out_dict.append(out)

        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """与原始UNTrack相同"""
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord_new = bbox.view(bs, Nq, 4)

            out = {
                'pred_boxes': outputs_coord_new,
                'score_map': score_map_ctr,
                'size_map': size_map,
                'offset_map': offset_map
            }
            return out
        else:
            raise NotImplementedError


# ==================== 模型构建函数 ====================

def build_untrack_pro(cfg, training=True):
    """
    构建改进版UNTrack模型

    保持与原始build_untrack相同的接口
    """
    import os
    from lib.models.layers.head import build_box_head
    from lib.models.untrack.vit_ce import vit_base_patch16_224_ce

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')

    # 加载预训练权重
    if cfg.MODEL.PRETRAIN_FILE and ('UNTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    # 构建骨干网络
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
        )
    else:
        raise NotImplementedError

    # 微调适配
    backbone.finetune_track(cfg=cfg, patch_start_index=1)

    # 构建预测头
    hidden_dim = backbone.embed_dim
    box_head = build_box_head(cfg, hidden_dim)

    # 构建改进版模型
    model = UNTrackPro(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=getattr(cfg.MODEL.BACKBONE, 'TOKEN_LEN', 1),
        num_searches=cfg.DATA.SEARCH.LENGTH,
        use_mshfc=getattr(cfg.MODEL, 'USE_MSHFC', True),
        use_lsmb=getattr(cfg.MODEL, 'USE_LSMB', True),
        use_dynamic_ce=getattr(cfg.MODEL, 'USE_DYNAMIC_CE', True),
    )

    return model


# 保持与原代码兼容
def build_untrack(cfg, training=True):
    """原始构建函数，可切换回标准版本"""
    use_pro = getattr(cfg.MODEL, 'USE_PRO', True)
    if use_pro:
        return build_untrack_pro(cfg, training)
    else:
        # 返回原始UNTrack（需要导入原始代码）
        from .untrack import build_untrack as build_original
        return build_original(cfg, training)

