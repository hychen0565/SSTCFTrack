import torch
from torch import nn
import torch.nn.functional as F
import math
import model.resnet as models
import model.vgg as vgg_models


# ===========================
# 1. 基础组件层（保留原实现，新增 RB 模块）
# ===========================

class GBC(nn.Module):
    """Gated Bottleneck Convolution - 保持 CGPANet 的轻量化核心"""

    def __init__(self, in_channels, out_channels=None, mid_ratio=8):
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = max(in_channels // mid_ratio, 16)
        self.pw1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.dw = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid_channels)
        self.pw2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.skip = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        g = self.gate(x)
        out = self.relu(self.bn1(self.pw1(x)))
        out = self.relu(self.bn_dw(self.dw(out)))
        out = self.bn2(self.pw2(out))
        return self.relu(residual + out * g)


class RB(nn.Module):
    """Residual Block with GroupNorm & SiLU - 来自 MFANet，提升小 Batch Size 下的稳定性"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels,
                                                                                kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class EMA(nn.Module):
    """Efficient Multi-scale Attention - 保持不变"""

    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, 1, bias=False)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, 3, padding=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class DySample(nn.Module):
    """Dynamic Upsampling - 保持不变"""

    def __init__(self, in_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, in_channels * (scale ** 2), kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.ps(self.conv(x)) + F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)


class PAF(nn.Module):
    """Pixel Attention Fusion - 保持不变"""

    def __init__(self, in_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or in_channels // 8
        self.transform_q = nn.Sequential(GBC(in_channels, mid_channels), nn.BatchNorm2d(mid_channels))
        self.transform_k = nn.Sequential(GBC(in_channels, mid_channels), nn.BatchNorm2d(mid_channels))
        self.adapter = nn.Sequential(GBC(mid_channels, in_channels), nn.Sigmoid())

    def forward(self, base_feat, guidance_feat):
        b, c, h, w = base_feat.shape
        if guidance_feat.shape[-2:] != (h, w):
            guidance_feat = F.interpolate(guidance_feat, (h, w), mode='bilinear', align_corners=False)
        q = self.transform_q(guidance_feat)
        k = self.transform_k(base_feat)
        sim = self.adapter(q * k)
        return (1 - sim) * base_feat + sim * guidance_feat


class MSCM(nn.Module):
    """Multi-Scale Context Module - 保持不变"""

    def __init__(self, d_model, reduction=16):
        super().__init__()
        self.local_attn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model))
        self.global_attn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                                         nn.Linear(d_model, d_model))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 4:
            b, c, h, w = x.shape
            x_flat = x.flatten(2).transpose(1, 2)
            pool = torch.mean(x_flat, dim=1, keepdim=True)
            attn = self.local_attn(x_flat) + self.global_attn(pool)
            attn = self.sigmoid(attn).transpose(1, 2).reshape(b, c, h, w)
            return x * attn
        else:
            pool = torch.mean(x, dim=1, keepdim=True)
            return x * self.sigmoid(self.local_attn(x) + self.global_attn(pool))


# ===========================
# 2. 核心创新模块 (融合 MFANet 优点)
# ===========================

class LDPG(nn.Module):
    """Lightweight Dynamic Prototype Generation (Enhanced)

    改进点:
    1. 保留 GBC+EMA 的轻量化多尺度提取。
    2. 引入 Peak Pooling (MFANet 核心)，专门捕获显著缺陷特征。
    3. 双路融合：AvgPool (背景上下文) + PeakPool (显著目标)。
    """

    def __init__(self, channels):
        super().__init__()
        self.inter_c = channels // 2
        self.gbc = GBC(channels, self.inter_c)
        self.theta = GBC(channels, self.inter_c)
        self.phi = GBC(channels, self.inter_c)
        self.ema = EMA(self.inter_c)

        # 投影层
        self.W = nn.Sequential(
            GBC(self.inter_c, channels),
            nn.BatchNorm2d(channels)
        )

        # 关键改进：引入 Peak Pooling 分支
        self.peak_pool = nn.AdaptiveMaxPool2d(1)  # 显著特征
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局上下文

        # 通道注意力调制
        self.mscm = MSCM(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b = x.size(0)

        # 1. 特征提取与EMA增强
        gbc_feat = self.gbc(x)
        ema_feat = self.ema(gbc_feat)

        # 2. 轻量注意力计算（保留原有的动态计算逻辑）
        theta_x = self.theta(x)
        phi_x = self.phi(x)
        g_flat = ema_feat.view(b, self.inter_c, -1).permute(0, 2, 1)
        theta_flat = theta_x.view(b, self.inter_c, -1).permute(0, 2, 1)
        phi_flat = phi_x.view(b, self.inter_c, -1)

        attn = F.softmax(torch.bmm(theta_flat, phi_flat), dim=-1)
        y = torch.bmm(attn, g_flat).permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_c, *x.shape[2:])

        # 3. 特征融合
        W_y = self.W(y)
        z = self.gamma * W_y + x

        # 4. 关键步骤：双统计量池化生成原型
        feat = self.mscm(z)
        vec_avg = self.global_pool(feat)  # 常规全局向量
        vec_peak = self.peak_pool(feat)  # 显著性向量 (关键改进)

        # 融合策略：简单相加，保留两者的优势
        proto_vec = vec_avg + vec_peak

        return proto_vec  # [B, C, 1, 1]


class AMGD(nn.Module):
    """Adaptive Multi-Granularity Decoder (Enhanced)

    改进点:
    1. 使用 RB (GroupNorm+SiLU) 替代部分 Conv+BN，提升训练稳定性。
    2. 保留 DySample 和 PAF 的动态融合优势。
    """

    def __init__(self, dim=256, drop_rate=0.5):
        super().__init__()
        self.ema = EMA(dim)

        # 融合层
        self.fusion_conv = nn.Sequential(
            GBC(dim * 2, dim),
            EMA(dim)
        )

        # 上采样模块
        self.up1 = nn.Sequential(GBC(dim, dim), DySample(dim, 2))
        self.up2 = nn.Sequential(GBC(dim, dim), DySample(dim, 2))

        # 多尺度卷积
        self.ms_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, padding=2 ** i, dilation=2 ** i, bias=False),
                nn.BatchNorm2d(dim // 4),
                nn.ReLU(inplace=True)
            ) for i in range(3)
        ])
        self.fusion = GBC(dim + 3 * (dim // 4), dim)

        # 融合 MFANet 的 RB 模块
        self.detail_branch = nn.Sequential(RB(dim, dim), EMA(dim), RB(dim, dim))
        self.semantic_branch = nn.Sequential(RB(dim, dim), RB(dim, dim))

        self.head = nn.Sequential(
            RB(dim, dim // 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(dim // 2, 2, 1, bias=False)
        )

    def forward(self, query_feat, support_feat, support_mask, merge_feat, h, w):
        # 1. 跨域融合
        sup_h, sup_w = query_feat.shape[2:]
        support_exp = support_feat.expand(-1, -1, sup_h, sup_w)
        fused_feat = self.fusion_conv(torch.cat([query_feat, support_exp], dim=1))
        fused_feat = self.ema(fused_feat)

        # 2. 解码路径
        x2 = self.up1(merge_feat)
        x4 = self.up2(x2)

        # 3. 多尺度特征聚合
        target_size = x4.shape[2:]
        x2_resized = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        fused_resized = F.interpolate(fused_feat, size=target_size, mode='bilinear', align_corners=False)

        # 融合低层与高层
        decode_feat = x4 + x2_resized + fused_resized

        # 多尺度上下文
        multi_feats = [decode_feat]
        for conv in self.ms_conv:
            multi_feats.append(conv(decode_feat))
        decode_feat = self.fusion(torch.cat(multi_feats, dim=1))

        # 4. 双分支 (RB 模块增强)
        out1 = self.detail_branch(decode_feat)
        out2 = self.semantic_branch(decode_feat)
        out = out1 + out2

        return self.head(out)


class GBC_ResBlock(nn.Module):
    """GBC残差块 - 用于特征增强"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(GBC(in_c, out_c), GBC(out_c, out_c))
        self.skip = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        return self.conv(x) + self.skip(x)


# ===========================
# 3. 主网络架构 (CGPANet-V2)
# ===========================

class CGPANet(nn.Module):
    """Cross-Granularity Prototype Aggregation Network V2

    关键升级:
    1. LDPG: 引入 Peak Pooling，增强显著特征提取。
    2. Decoder: 引入 GroupNorm+SiLU，提升训练稳定性。
    3. 保留 SPM Prior: 维持显式定位能力。
    """

    def __init__(self, args):
        super().__init__()
        from torch.nn import BatchNorm2d as BatchNorm

        # Backbone 初始化 (保持不变)
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.shot = args.shot
        self.vgg = args.vgg
        self.classes = args.classes
        self.pretrained = True
        models.BatchNorm = BatchNorm
        self.layers = args.layers

        if self.vgg:
            print('>>>>>>>>> Using VGG_16 bn <<<<<<<<<')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = self._get_vgg16_layer(vgg16)
        else:
            print(f'>>>>>>>>> Using ResNet {self.layers} <<<<<<<<<')
            if self.layers == 50:
                resnet = models.resnet50(pretrained=self.pretrained)
            elif self.layers == 101:
                resnet = models.resnet101(pretrained=self.pretrained)

            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = \
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            # 空洞卷积
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        fea_dim = 512 + 256 if self.vgg else 1024 + 512

        # 1. 特征嵌入
        self.down_query = nn.Sequential(GBC(fea_dim, reduce_dim), nn.GroupNorm(32, reduce_dim), nn.SiLU())
        self.down_supp = nn.Sequential(GBC(fea_dim, reduce_dim), nn.GroupNorm(32, reduce_dim), nn.SiLU())

        # 2. 核心模块: LDPG (含 Peak Pooling)
        self.ldpg = LDPG(reduce_dim)

        # 3. 解码器: AMGD (含 GroupNorm)
        self.amgd = AMGD(reduce_dim)

        # 4. 融合模块
        self.paf_ref = PAF(reduce_dim)
        self.paf_att = PAF(reduce_dim)

        self.init_merge = nn.Sequential(
            GBC(reduce_dim * 3 + 1, reduce_dim),  # 增加 1 个通道给 CPM 输出
            nn.GroupNorm(32, reduce_dim),
            nn.SiLU()
        )

        self.feature_enhance = nn.Sequential(
            GBC_ResBlock(reduce_dim, reduce_dim),
            EMA(reduce_dim),
            GBC_ResBlock(reduce_dim, reduce_dim)
        )

        # 辅助分支
        simple_in = 512 if self.vgg else 2048
        self.simple_proj = nn.Sequential(
            nn.Conv2d(simple_in, reduce_dim, 1, bias=False),
            nn.GroupNorm(32, reduce_dim),
            nn.SiLU()
        )
        self.simple_enhance = nn.Sequential(RB(reduce_dim, reduce_dim), RB(reduce_dim, reduce_dim))
        self.simple_head = nn.Conv2d(reduce_dim, self.classes, 1)

        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self._freeze_backbone()

    def _freeze_backbone(self):
        for m in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in m.parameters():
                p.requires_grad = False

    def _get_vgg16_layer(self, model):
        ranges = [range(0, 7), range(7, 14), range(14, 24), range(24, 34), range(34, 43)]
        return tuple(nn.Sequential(*[model.features[i] for i in r]) for r in ranges)

    def get_optim(self, args, LR):
        gbc_params, other_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ['dw', 'pw1', 'pw2', 'gate']):
                gbc_params.append(p)
            else:
                other_params.append(p)

        param_groups = [{'params': other_params, 'lr': LR}]
        if gbc_params:
            param_groups.append({'params': gbc_params, 'lr': LR * 0.5})

        return torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)

    def _compute_spm_prior(self, q4, supp_list, mask_list, target_size):
        """计算 SPM 先验掩码 (保留原逻辑)"""
        corr_list, eps = [], 1e-7
        for i, supp_feat in enumerate(supp_list):
            size = supp_feat.size(2)
            mask = F.interpolate(mask_list[i], size=(size, size), mode='bilinear', align_corners=True)
            masked_supp = supp_feat * mask
            b, c, sp = q4.size(0), q4.size(1), size * size
            q = q4.view(b, c, -1)
            s = masked_supp.view(b, c, -1).permute(0, 2, 1)
            q_norm = torch.norm(q, 2, 1, True)
            s_norm = torch.norm(s, 2, 2, True)
            sim = torch.bmm(s, q) / (torch.bmm(s_norm, q_norm) + eps)
            sim = sim.max(1)[0].view(b, sp)
            sim = (sim - sim.min(1, keepdim=True)[0]) / \
                  (sim.max(1, keepdim=True)[0] - sim.min(1, keepdim=True)[0] + eps)
            corr = sim.view(b, 1, size, size)
            corr = F.interpolate(corr, size=target_size, mode='bilinear', align_corners=True)
            corr_list.append(corr)
        corr = torch.stack(corr_list).mean(dim=0)
        return corr

    def forward(self, x, s_x=None, s_y=None, y=None):
        if s_x is None:
            s_x = torch.zeros(x.size(0), self.shot, 3, x.size(2), x.size(3), device=x.device)
            s_y = torch.zeros(x.size(0), self.shot, x.size(2), x.size(3), device=x.device)

        h, w = x.size()[2:]

        # 1. Backbone 特征提取
        with torch.no_grad():
            q0 = self.layer0(x)
            q1 = self.layer1(q0)
            q2 = self.layer2(q1)
            q3 = self.layer3(q2)
            q4 = self.layer4(q3)
            if self.vgg:
                q2 = F.interpolate(q2, size=q3.shape[2:], mode='bilinear', align_corners=True)
            query_cat = torch.cat([q3, q2], dim=1)

        query_feat = self.down_query(query_cat)

        mask_list, proto_list, att_list, supp_high_list = [], [], [], []

        for i in range(self.shot):
            supp_gt = (s_y[:, i] == 1).float().unsqueeze(1)

            with torch.no_grad():
                s0 = self.layer0(s_x[:, i])
                s1 = self.layer1(s0)
                s2 = self.layer2(s1)
                s3 = self.layer3(s2)
                s4 = self.layer4(s3)
                if self.vgg:
                    s2 = F.interpolate(s2, size=s3.shape[2:], mode='bilinear', align_corners=True)
                mask = F.interpolate(supp_gt, size=s3.shape[2:], mode='bilinear', align_corners=True)
                supp_cat = torch.cat([s3, s2], dim=1)

                supp_high_list.append(s4)
                mask_list.append(supp_gt)

            # Support 投影
            supp_feat = self.down_supp(supp_cat)

            # 核心步骤 1: LDPG (含 Peak Pooling)
            prototype = self.ldpg(supp_feat * mask)
            proto_list.append(prototype)

            # 核心步骤 2: Support Attention (保持不变)
            supp_att = self.paf_att(supp_feat, supp_feat * mask)
            att_list.append(supp_att)

        # K-shot 聚合
        prototype = torch.stack(proto_list).mean(dim=0) if self.shot > 1 else proto_list[0]
        supp_att = torch.stack(att_list).mean(dim=0) if self.shot > 1 else att_list[0]

        # 核心步骤 3: SPM Prior (显式相关性先验 - MFANet 优势迁移)
        corr_mask = self._compute_spm_prior(q4, supp_high_list, mask_list, query_feat.shape[2:])

        # 特征融合 (增加了 corr_mask 通道)
        merge_feat = self.init_merge(torch.cat([query_feat, prototype, supp_att, corr_mask], dim=1))
        merge_feat = self.feature_enhance(merge_feat)

        # 解码
        seg_out = self.amgd(
            query_feat=query_feat,
            support_feat=prototype,
            support_mask=supp_gt,
            merge_feat=merge_feat,
            h=h, w=w
        )

        if seg_out.shape[2:] != (h, w):
            seg_out = F.interpolate(seg_out, size=(h, w), mode='bilinear', align_corners=True)

        # 辅助分支
        simple_feat = self.simple_proj(q4)
        simple_feat = self.simple_enhance(simple_feat)
        simple_out = self.simple_head(simple_feat)
        simple_out = F.interpolate(simple_out, size=(h, w), mode='bilinear', align_corners=True)

        if not self.training:
            seg_out = self.max_pool(seg_out)
            seg_out = -self.max_pool(-seg_out)
            return seg_out

        # Loss 计算
        main_loss = self.criterion(seg_out, y.long())
        aux_loss = self.criterion(simple_out, y.long())
        total_loss = main_loss + 0.4 * aux_loss

        return seg_out.max(1)[1], total_loss


if __name__ == '__main__':
    import os
    import argparse

    # ==========================================
    # 1. 配置模拟参数
    # ==========================================
    class Args:
        def __init__(self):
            self.shot = 5  # 1-shot setting
            self.vgg = False  # 使用 ResNet50
            self.layers = 50  # ResNet50
            self.classes = 2  # 类别数 (背景 + 缺陷)
            self.ignore_label = 255  # 忽略标签
            self.momentum = 0.9
            self.weight_decay = 0.0005
            self.base_lr = 0.001


    args = Args()

    # 为了快速测试结构，避免下载预训练权重，此处临时修改
    # 实际训练时请保留模型内部的 self.pretrained = True
    print(">>> Initializing model for testing (pretrained=False for speed)...")

    # ==========================================
    # 2. 实例化模型
    # ==========================================
    model = CGPANet(args)

    # 设置为 CPU 测试 (如有 GPU 可改为 cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # ==========================================
    # 3. 模型参数统计
    # ==========================================
    # def count_parameters(model, table_flag=True):
    #     total_params = 0
    #     for name, parameter in model.named_parameters():
    #         if not parameter.requires_grad:
    #             continue
    #         param = parameter.numel()
    #         # table.add_row([name, param])
    #         total_params += param
    #     if table_flag:
    #         print(table)
    #     return total_params

    print("Model Parameters Statistics")
    print("=" * 30)
    # 计算总参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算核心模块参数
    ldpg_params = sum(p.numel() for p in model.ldpg.parameters())
    amgd_params = sum(p.numel() for p in model.amgd.parameters())

    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
    print(f"LDPG Module Parameters: {ldpg_params / 1e6:.2f} M ({ldpg_params / total_params * 100:.2f}%)")
    print(f"AMGD Module Parameters: {amgd_params / 1e6:.2f} M ({amgd_params / total_params * 100:.2f}%)")
    print(f"Backbone (Frozen) Parameters: {(total_params - trainable_params) / 1e6:.2f} M")

    # ==========================================
    # 4. 输入数据模拟

    print("Forward Pass Testing")

    batch_size = 2
    h, w = 256, 256  # 测试分辨率

    # 模拟输入
    x_query = torch.randn(batch_size, 3, h, w).to(device)  # Query Image
    s_x = torch.randn(batch_size, args.shot, 3, h, w).to(device)  # Support Image
    s_y = torch.randint(0, 2, (batch_size, args.shot, h, w)).to(device)  # Support Mask
    y_query = torch.randint(0, args.classes, (batch_size, h, w)).to(device)  # Query Mask (for loss)

    print(f"Input Query Shape: {x_query.shape}")
    print(f"Input Support Shape: {s_x.shape}")
    print(f"Input Support Mask Shape: {s_y.shape}")

    # ==========================================
    # 5. Train 模式测试
    # ==========================================
    model.train()
    print("\n[Test] Running in TRAIN mode...")

    try:
        # 前向传播
        pred, loss = model(x_query, s_x, s_y, y_query)

        print(f"Output Pred Shape: {pred.shape}")  # 应为 [B, H, W]
        print(f"Loss Value: {loss.item():.4f}")

        # 反向传播测试
        loss.backward()
        print("Backward pass: SUCCESS")

    except Exception as e:
        print(f"Forward/Backward Failed: {e}")

    # ==========================================
    # 6. Eval 模式测试
    # ==========================================
    model.eval()
    print("\n[Test] Running in EVAL mode...")
    with torch.no_grad():
        try:
            # Eval 模式不需要传入 y
            output = model(x_query, s_x, s_y)
            print(f"Output Segmentation Shape: {output.shape}")  # 应为 [B, H, W]
            print("Eval Forward pass: SUCCESS")

            # 检查输出范围
            print(f"Output Min: {output.min().item():.4f}, Max: {output.max().item():.4f}")

        except Exception as e:
            print(f"Eval Forward Failed: {e}")
    print("\n>>> Test Completed.")
