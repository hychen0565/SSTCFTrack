import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None, add_cls_token=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.add_cls_token = add_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    # # 补充 finetune_track 方法（根据项目需求实现）
    # def finetune_track(self, x, template=None, **kwargs):
    #     # 示例逻辑：跟踪任务的前向传播
    #     features = self.backbone(x, template=template)
    #     output = self.head(features)
    #     return output

    def forward_features(self, z, xs, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, track_query=None,
                         token_type="add", token_len=1
                         ):
        B, H, W = xs[-1].shape[0], xs[-1].shape[2], xs[-1].shape[3]
        num_searches = len(xs)

        x = self.patch_embed(xs[-1])
        for ind in range(num_searches-1, 0, -1):
            x_ = self.patch_embed(xs[ind-1])
            x = torch.cat((x_,x), dim=1)
        top_k_indices = None

        z = torch.stack(z, dim=1)
        _, T_z, C_z, H_z, W_z = z.shape
        z = z.flatten(0, 1)
        z = self.patch_embed(z)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            if token_type == "concat":
                if track_query is None:
                    query = self.cls_token.expand(B, token_len, -1)
                else:
                    track_len = track_query.size(1)
                    new_query = self.cls_token.expand(B, token_len - track_len, -1)
                    query = torch.cat([new_query, track_query], dim=1)
            elif token_type == "add":
                new_query = self.cls_token.expand(B, token_len, -1)  # copy B times
                query = new_query if track_query is None else track_query + new_query
            query = query + self.cls_pos_embed

        z = z + self.pos_embed_z
        x = x + self.pos_embed_x

        if self.add_sep_seg:
            x = x + self.search_segment_pos_embed
            z = z + self.template_segment_pos_embed

        if T_z > 1:  # multiple memory frames
            z = z.view(B, T_z, -1, z.size()[-1]).contiguous()
            z = z.flatten(1, 2)

        lens_z = z.shape[1]  # HW
        lens_x = x.shape[1]  # HW

        x = combine_tokens(z, x, mode=self.cat_mode)  # (B, z+x, 768)
        if self.add_cls_token:
            x = torch.cat([query, x], dim=1)     # (B, 1+z+x, 768)
            query_len = query.size(1)
        x = self.pos_drop(x)

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []
        for i, blk in enumerate(self.blocks):
            if self.add_cls_token:
                x, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
                        add_cls_token=self.add_cls_token, query_len=query_len, lens_z=lens_z, lens_x=lens_x)
            else:
                x, global_index_t, global_index_s, removed_index_s, attn = \
                    blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate, add_cls_token=self.add_cls_token)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        if self.add_cls_token:
            query = x[:, :query_len]
            z = x[:, query_len:lens_z_new+query_len]
            x = x[:, lens_z_new+query_len:]
        else:
            z = x[:, :lens_z_new]
            x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        if self.add_cls_token:
            x = torch.cat([query, z, x], dim=1)
        else:
            x = torch.cat([z, x], dim=1)

        # aux_dict = {}
        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict, top_k_indices

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, return_last_attn=False, track_query=None,
                token_type="add", token_len=1):
        x, aux_dict, top_k_indices = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                            track_query=track_query, token_type=token_type, token_len=token_len)
        return x, aux_dict, top_k_indices


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            # try:
            checkpoint = torch.load(pretrained, map_location="cpu",weights_only=False)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print('Load pretrained model from: ' + pretrained)
            # except:
            #     print("Warning: MAE Pretrained model weights are not loaded !")

    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, in_chans=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp
#
#
# # 补全缺失的辅助函数（根据代码上下文推测）
# def combine_tokens(z, x, mode='direct'):
#     if mode == 'direct':
#         return torch.cat([z, x], dim=1)
#     else:
#         raise NotImplementedError(f"Combine mode {mode} not supported")
#
#
# def recover_tokens(x, lens_z_new, lens_x, mode='direct'):
#     if mode == 'direct':
#         return x
#     else:
#         raise NotImplementedError(f"Recover mode {mode} not supported")
#
#
# class Block(nn.Module):
#     """简化的Transformer Block（适配你的代码逻辑）"""
#
#     def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0.):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
#
#     def forward(self, x, global_index_t, global_index_s, mask_x=None, ce_template_mask=None,
#                 ce_keep_rate=None, add_cls_token=True, query_len=1, lens_z=0, lens_x=0):
#         # 简化的前向逻辑（适配你的返回值要求）
#         attn_output, attn_weights = self.attn(self.norm1(x), x, x, key_padding_mask=mask_x)
#         x = x + self.drop_path(attn_output)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x, global_index_t, global_index_s, None, attn_weights
#
#
# class VisionTransformerCE(nn.Module):
#     def __init__(self, patch_size=16, in_chans=8, embed_dim=768, depth=12, num_heads=12,
#                  add_cls_token=True, cls_token_use_mode='ignore', cat_mode='direct',
#                  ce_loc=[3, 6, 9], ce_keep_ratio=[0.7, 0.7, 0.7], sep_seg=False):
#         super().__init__()
#         # 基础配置
#         self.patch_size = patch_size
#         self.add_cls_token = add_cls_token
#         self.cls_token_use_mode = cls_token_use_mode
#         self.cat_mode = cat_mode
#         self.ce_loc = ce_loc
#         self.ce_keep_ratio = ce_keep_ratio
#         self.sep_seg = sep_seg
#         self.drop_path_rate = drop_path_rate
#
#         # Patch Embedding
#         self.patch_embed = PatchEmbed(
#             img_size=224, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#
#         # CLS Token
#         if add_cls_token:
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#             self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
#
#         # 位置编码
#         self.pos_embed_z = nn.Parameter(torch.zeros(1, (224 // patch_size) ** 2, embed_dim))
#         self.pos_embed_x = nn.Parameter(torch.zeros(1, (224 // patch_size) ** 2, embed_dim))
#         if sep_seg:
#             self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, (224 // patch_size) ** 2, embed_dim))
#             self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, (224 // patch_size) ** 2, embed_dim))
#
#         self.pos_drop = nn.Dropout(p=0.)
#
#         # Transformer Blocks
#         self.blocks = nn.ModuleList([
#             Block(embed_dim, num_heads) for _ in range(depth)
#         ])
#
#         self.norm = nn.LayerNorm(embed_dim)
#
#     def forward_features(self, z, x, ce_template_mask=None, ce_keep_rate=None,
#                          track_query=None, token_type="add", token_len=1):
#         B = z.shape[0]
#         T_z = z.shape[1] if len(z.shape) == 5 else 1
#
#         # 展平并嵌入patch
#         if len(z.shape) == 5:
#             z = z.flatten(0, 1)
#         z = self.patch_embed(z)
#
#         x = self.patch_embed(x)
#         mask_z = None
#         mask_x = None
#
#         # attention mask handling
#         if mask_z is not None and mask_x is not None:
#             mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
#             mask_z = mask_z.flatten(1).unsqueeze(-1)
#
#             mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
#             mask_x = mask_x.flatten(1).unsqueeze(-1)
#
#             mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
#             mask_x = mask_x.squeeze(-1)
#
#         # CLS Token处理
#         if self.add_cls_token:
#             if token_type == "concat":
#                 if track_query is None:
#                     query = self.cls_token.expand(B, token_len, -1)
#                 else:
#                     track_len = track_query.size(1)
#                     new_query = self.cls_token.expand(B, token_len - track_len, -1)
#                     query = torch.cat([new_query, track_query], dim=1)
#             elif token_type == "add":
#                 new_query = self.cls_token.expand(B, token_len, -1)
#                 query = new_query if track_query is None else track_query + new_query
#             query = query + self.cls_pos_embed
#
#         # 位置编码
#         z = z + self.pos_embed_z
#         x = x + self.pos_embed_x
#
#         if self.add_sep_seg:
#             x = x + self.search_segment_pos_embed
#             z = z + self.template_segment_pos_embed
#
#         if T_z > 1:
#             z = z.view(B, T_z, -1, z.size()[-1]).contiguous()
#             z = z.flatten(1, 2)
#
#         lens_z = z.shape[1]
#         lens_x = x.shape[1]
#
#         x = combine_tokens(z, x, mode=self.cat_mode)
#         if self.add_cls_token:
#             x = torch.cat([query, x], dim=1)
#             query_len = query.size(1)
#         x = self.pos_drop(x)
#
#         # 全局索引
#         global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
#         global_index_t = global_index_t.repeat(B, 1)
#         global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
#         global_index_s = global_index_s.repeat(B, 1)
#
#         removed_indexes_s = []
#         top_k_indices = None  # 补全缺失的返回值
#
#         # 遍历Transformer Blocks
#         for i, blk in enumerate(self.blocks):
#             if self.add_cls_token:
#                 x, global_index_t, global_index_s, removed_index_s, attn = \
#                     blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
#                         add_cls_token=self.add_cls_token, query_len=query_len, lens_z=lens_z, lens_x=lens_x)
#             else:
#                 x, global_index_t, global_index_s, removed_index_s, attn = \
#                     blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate,
#                         add_cls_token=self.add_cls_token)
#
#             if self.ce_loc is not None and i in self.ce_loc:
#                 removed_indexes_s.append(removed_index_s)
#
#         x = self.norm(x)
#         lens_x_new = global_index_s.shape[1]
#         lens_z_new = global_index_t.shape[1]
#
#         # 拆分token
#         if self.add_cls_token:
#             query = x[:, :query_len]
#             z = x[:, query_len:lens_z_new + query_len]
#             x = x[:, lens_z_new + query_len:]
#         else:
#             z = x[:, :lens_z_new]
#             x = x[:, lens_z_new:]
#
#         # 恢复token
#         if removed_indexes_s and removed_indexes_s[0] is not None:
#             removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
#             pruned_lens_x = lens_x - lens_x_new
#             pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
#             x = torch.cat([x, pad_x], dim=1)
#             index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
#             C = x.shape[-1]
#             x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
#                                              src=x)
#
#         x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
#
#         # 重新拼接
#         if self.add_cls_token:
#             x = torch.cat([query, z, x], dim=1)
#         else:
#             x = torch.cat([z, x], dim=1)
#
#         aux_dict = {
#             "attn": attn,
#             "removed_indexes_s": removed_indexes_s,
#         }
#
#         return x, aux_dict, top_k_indices
#
#     def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
#                 tnc_keep_rate=None, return_last_attn=False, track_query=None,
#                 token_type="add", token_len=1):
#         x, aux_dict, top_k_indices = self.forward_features(z, x, ce_template_mask=ce_template_mask,
#                                                            ce_keep_rate=ce_keep_rate, track_query=track_query,
#                                                            token_type=token_type, token_len=token_len)
#         return x, aux_dict, top_k_indices
#
#     # ========== 关键修复：补全finetune_track方法 ==========
#     def finetune_track(self, z, x, ce_template_mask=None, ce_keep_rate=None,
#                        track_query=None, token_type="add", token_len=1):
#         """
#         追踪微调的前向方法（补全缺失的方法定义）
#         参数说明：
#         - z: template特征
#         - x: search区域特征（必需参数，修复参数缺失问题）
#         - ce_template_mask: CE模板掩码
#         - ce_keep_rate: CE保留率
#         - track_query: 追踪查询token
#         - token_type: token融合方式（add/concat）
#         - token_len: token长度
#         """
#         # 复用forward逻辑，适配追踪场景
#         return self.forward(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
#                             track_query=track_query, token_type=token_type, token_len=token_len)
#
#
# # 模型创建函数（保持不变）
# def _create_vision_transformer(pretrained=False, **kwargs):
#     model = VisionTransformerCE(**kwargs)
#
#     if pretrained:
#         if 'npz' in str(pretrained):
#             model.load_pretrained(pretrained, prefix='')
#         else:
#             try:
#                 checkpoint = torch.load(pretrained, map_location="cpu", weights_only=False)
#                 missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
#                 print("missing keys:", missing_keys)
#                 print("unexpected keys:", unexpected_keys)
#                 print('Load pretrained model from: ' + str(pretrained))
#             except Exception as e:
#                 print(f"Warning: MAE Pretrained model weights are not loaded ! Error: {e}")
#
#     return model
#
#
# def vit_base_patch16_224_ce(pretrained=False, **kwargs):
#     model_kwargs = dict(
#         patch_size=16, in_chans=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
#     return model