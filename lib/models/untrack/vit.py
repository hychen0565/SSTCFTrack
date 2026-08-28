import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

# from timm.models import helpers  # 或者直接导入需要的函数
# from timm.layers import Mlp, DropPath, trunc_normal_, lecun_normal_, to_2tuple
# from timm.models import registry
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_, to_2tuple
from timm.models.registry import register_model

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.untrack.base_backbone import BaseBackbone

from ..layers.attn_blocks import Block


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attn_type="concat"):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.attn_type = attn_type

#     def forward(self, x, lens_z, lens_x, return_attention=False):
#         if self.attn_type == 'concat':
#             B, N, C = x.shape
#             qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)

#             x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#             x = self.proj(x)
#             x = self.proj_drop(x)
#         elif self.attn_type == 'separate':
#             B, N, C = x.shape
#             qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#             q_track, q_t, q_s = torch.split(q, [1, lens_z, lens_x], dim=2)
#             t_track, k_t, k_s = torch.split(k, [1, lens_z, lens_x], dim=2)
#             v_track, v_t, v_s = torch.split(v, [1, lens_z, lens_x], dim=2)

#             # template attention
#             attn = (q_t @ k_t.transpose(-2, -1)) * self.scale  # (B, head, N_q, N)
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x_t = rearrange(attn @ v_t, 'b h t d -> b t (h d)')

#             # search region attention
#             k_ts = torch.cat([k_t, k_s], dim=2)
#             v_ts = torch.cat([v_t, v_s], dim=2)
#             attn = (q_s @ k_ts.transpose(-2, -1)) * self.scale  # (B, head, N_s, N)
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x_s = rearrange(attn @ v_ts, 'b h t d -> b t (h d)')

#             # track_query attention
#             attn = (q_track @ k.transpose(-2, -1)) * self.scale  # (B, head, N_s, N)
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x_track = rearrange(attn @ v, 'b h t d -> b t (h d)')

#             x = torch.cat([x_track, x_t, x_s], dim=1)

#             x = self.proj(x)
#             x = self.proj_drop(x)

#         if return_attention:
#             return x, attn
#         return x


# class Block(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type="concat"):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, attn_type=attn_type)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x, lens_z, lens_x, return_attention=False):
#         if return_attention:
#             feat, attn = self.attn(self.norm1(x), lens_z, lens_x, True)
#             x = x + self.drop_path(feat)
#             x = x + self.drop_path(self.mlp(self.norm2(x)))
#             return x, attn
#         else:
#             x = x + self.drop_path(self.attn(self.norm1(x), lens_z, lens_x))
#             x = x + self.drop_path(self.mlp(self.norm2(x)))
#             return x


class VisionTransformer(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', add_cls_token=False, attn_type="concat"):
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
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # # Representation layer
        # if representation_size and not distilled:
        #     self.num_features = representation_size
        #     self.pre_logits = nn.Sequential(OrderedDict([
        #         ('fc', nn.Linear(embed_dim, representation_size)),
        #         ('act', nn.Tanh())
        #     ]))
        # else:
        #     self.pre_logits = nn.Identity()
        #
        # # Classifier head(s)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head_dist = None
        # if distilled:
        #     self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token', 'dist_token'}
        return {'pos_embed', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = VisionTransformer(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            try:
                checkpoint = torch.load(pretrained, map_location="cpu",weights_only=False)
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
                print("missing keys:", missing_keys)
                print("unexpected keys:", unexpected_keys)
                print('Load pretrained model from: ' + pretrained)
            except:
                print("Warning: MAE Pretrained model weights are not loaded !")

    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, in_chans=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
# import math
# import logging
# from functools import partial
# from collections import OrderedDict
# from copy import deepcopy
# from einops import rearrange
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # ========== 修复 timm 导入 ==========
# # 新的导入路径（timm >= 0.6.0）
# # from timm import models  # 推荐：统一从 timm.models 导入
# # from timm.layers import Mlp, DropPath, trunc_normal_, lecun_normal_, to_2tuple  # layers 从 timm.layers 导入
# # from timm.models import register_model  # registry 从 timm.models 导入
# # from timm.models import load_checkpoint  # helpers 从 timm.models 导入
# from timm.models import helpers
# from timm.layers import Mlp, DropPath, trunc_normal_, lecun_normal_, to_2tuple
# from timm.models import registry
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
#
#
# # ========== 模拟缺失的自定义类（如果实际项目中有这些文件，替换为真实导入） ==========
# # 如果你的项目中有 lib/models/layers/patch_embed.py，请替换为：
# # from lib.models.layers.patch_embed import PatchEmbed
# # from lib.models.untrack.base_backbone import BaseBackbone
# # from ..layers.attn_blocks import Block
#
# # 临时模拟实现（保证代码能运行）
# class BaseBackbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#
# class PatchEmbed(nn.Module):
#     """ 简单的Patch Embedding实现 """
#
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches
#
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x
#
#
# # ========== 补充 Block 类（原代码中注释掉了，需要恢复） ==========
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attn_type="concat"):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.attn_type = attn_type
#
#     def forward(self, x, lens_z=0, lens_x=0, return_attention=False):
#         if self.attn_type == 'concat':
#             B, N, C = x.shape
#             qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             q, k, v = qkv[0], qkv[1], qkv[2]
#
#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#
#             x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#             x = self.proj(x)
#             x = self.proj_drop(x)
#         elif self.attn_type == 'separate':
#             B, N, C = x.shape
#             qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             q, k, v = qkv[0], qkv[1], qkv[2]
#
#             q_track, q_t, q_s = torch.split(q, [1, lens_z, lens_x], dim=2)
#             t_track, k_t, k_s = torch.split(k, [1, lens_z, lens_x], dim=2)
#             v_track, v_t, v_s = torch.split(v, [1, lens_z, lens_x], dim=2)
#
#             # template attention
#             attn = (q_t @ k_t.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x_t = rearrange(attn @ v_t, 'b h t d -> b t (h d)')
#
#             # search region attention
#             k_ts = torch.cat([k_t, k_s], dim=2)
#             v_ts = torch.cat([v_t, v_s], dim=2)
#             attn = (q_s @ k_ts.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x_s = rearrange(attn @ v_ts, 'b h t d -> b t (h d)')
#
#             # track_query attention
#             attn = (q_track @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x_track = rearrange(attn @ v, 'b h t d -> b t (h d)')
#
#             x = torch.cat([x_track, x_t, x_s], dim=1)
#             x = self.proj(x)
#             x = self.proj_drop(x)
#
#         if return_attention:
#             return x, attn
#         return x
#
#
# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type="concat"):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
#                               attn_type=attn_type)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#     def forward(self, x, lens_z=0, lens_x=0, return_attention=False):
#         if return_attention:
#             feat, attn = self.attn(self.norm1(x), lens_z, lens_x, True)
#             x = x + self.drop_path(feat)
#             x = x + self.drop_path(self.mlp(self.norm2(x)))
#             return x, attn
#         else:
#             x = x + self.drop_path(self.attn(self.norm1(x), lens_z, lens_x))
#             x = x + self.drop_path(self.mlp(self.norm2(x)))
#             return x
#
#
# # ========== VisionTransformer 主类（修复后） ==========
# class VisionTransformer(BaseBackbone):
#     """ Vision Transformer 实现 """
#
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
#                  act_layer=None, weight_init='', add_cls_token=False, attn_type="concat"):
#         super().__init__()
#         if isinstance(img_size, tuple):
#             self.img_size = img_size
#         else:
#             self.img_size = to_2tuple(img_size)
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim
#         self.num_tokens = 2 if distilled else 1
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         act_layer = act_layer or nn.GELU
#
#         self.patch_embed = embed_layer(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches
#
#         self.add_cls_token = add_cls_token
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         self.blocks = nn.Sequential(*[
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
#                 attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
#                 attn_type=attn_type)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)
#
#         self.init_weights(weight_init)
#
#     # 补充 finetune_track 方法（根据项目需求实现）
#     def finetune_track(self, x, template=None, **kwargs):
#         # 示例逻辑：跟踪任务的前向传播
#         features = self.backbone(x, template=template)
#         output = self.head(features)
#         return output
#
#     def init_weights(self, mode=''):
#         assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
#         head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
#         trunc_normal_(self.pos_embed, std=.02)
#         if self.dist_token is not None:
#             trunc_normal_(self.dist_token, std=.02)
#         if mode.startswith('jax'):
#             named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
#         else:
#             trunc_normal_(self.cls_token, std=.02)
#             self.apply(_init_vit_weights)
#
#     def _init_weights(self, m):
#         _init_vit_weights(m)
#
#     @torch.jit.ignore()
#     def load_pretrained(self, checkpoint_path, prefix=''):
#         _load_weights(self, checkpoint_path, prefix)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'dist_token'}
#
#     def get_classifier(self):
#         if self.dist_token is None:
#             return self.head if hasattr(self, 'head') else nn.Identity()
#         else:
#             return (self.head, self.head_dist) if hasattr(self, 'head') else (nn.Identity(), nn.Identity())
#
#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
#         if self.num_tokens == 2:
#             self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
#
#     def forward(self, x):
#         """ 补充forward方法，保证模型可运行 """
#         x = self.patch_embed(x)
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)
#         if self.dist_token is not None:
#             x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
#         else:
#             x = torch.cat((cls_token, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
#
#         for blk in self.blocks:
#             x = blk(x)
#
#         x = self.norm(x)
#         return x
#
#
# # ========== 辅助函数（完整实现） ==========
# def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
#     """ ViT权重初始化 """
#     if isinstance(module, nn.Linear):
#         if name.startswith('head'):
#             nn.init.zeros_(module.weight)
#             nn.init.constant_(module.bias, head_bias)
#         elif name.startswith('pre_logits'):
#             lecun_normal_(module.weight)
#             nn.init.zeros_(module.bias)
#         else:
#             if jax_impl:
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     if 'mlp' in name:
#                         nn.init.normal_(module.bias, std=1e-6)
#                     else:
#                         nn.init.zeros_(module.bias)
#             else:
#                 trunc_normal_(module.weight, std=.02)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#     elif jax_impl and isinstance(module, nn.Conv2d):
#         lecun_normal_(module.weight)
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)
#     elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
#         nn.init.zeros_(module.bias)
#         nn.init.ones_(module.weight)
#
#
# @torch.no_grad()
# def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
#     """ 加载预训练权重 """
#     import numpy as np
#
#     def _n2p(w, t=True):
#         if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
#             w = w.flatten()
#         if t:
#             if w.ndim == 4:
#                 w = w.transpose([3, 2, 0, 1])
#             elif w.ndim == 3:
#                 w = w.transpose([2, 0, 1])
#             elif w.ndim == 2:
#                 w = w.transpose([1, 0])
#         return torch.from_numpy(w)
#
#     try:
#         w = np.load(checkpoint_path)
#     except:
#         print(f"无法加载预训练文件: {checkpoint_path}")
#         return
#
#     if not prefix and 'opt/target/embedding/kernel' in w:
#         prefix = 'opt/target/'
#
#     if hasattr(model.patch_embed, 'backbone'):
#         backbone = model.patch_embed.backbone
#         stem_only = not hasattr(backbone, 'stem')
#         stem = backbone if stem_only else backbone.stem
#         stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
#         stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
#         stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
#         if not stem_only:
#             for i, stage in enumerate(backbone.stages):
#                 for j, block in enumerate(stage.blocks):
#                     bp = f'{prefix}block{i + 1}/unit{j + 1}/'
#                     for r in range(3):
#                         getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
#                         getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
#                         getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
#                     if block.downsample is not None:
#                         block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
#                         block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
#                         block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
#         embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
#     else:
#         embed_conv_w = adapt_input_conv(
#             model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
#
#     model.patch_embed.proj.weight.copy_(embed_conv_w)
#     if model.patch_embed.proj.bias is not None:
#         model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
#
#     model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
#     pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
#
#     if pos_embed_w.shape != model.pos_embed.shape:
#         pos_embed_w = resize_pos_embed(
#             pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
#     model.pos_embed.copy_(pos_embed_w)
#     model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
#     model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
#
#     # 加载block权重
#     for i, block in enumerate(model.blocks.children()):
#         block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
#         mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
#         try:
#             block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
#             block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
#             block.attn.qkv.weight.copy_(torch.cat([
#                 _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
#             block.attn.qkv.bias.copy_(torch.cat([
#                 _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
#             block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
#             block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
#             for r in range(2):
#                 getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
#                 getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
#             block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
#             block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))
#         except KeyError as e:
#             print(f"加载block {i} 权重时缺失键: {e}")
#             continue
#
#
# def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
#     """ 调整位置编码大小 """
#     print(f'调整位置编码尺寸: {posemb.shape} -> {posemb_new.shape}')
#     ntok_new = posemb_new.shape[1]
#     if num_tokens:
#         posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
#         ntok_new -= num_tokens
#     else:
#         posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
#
#     gs_old = int(math.sqrt(len(posemb_grid)))
#     if not len(gs_new):
#         gs_new = [int(math.sqrt(ntok_new))] * 2
#     assert len(gs_new) >= 2
#     print(f'位置编码网格大小: [{gs_old}, {gs_old}] -> {gs_new}')
#
#     posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
#     posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear', align_corners=False)
#     posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
#     posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
#     return posemb
#
#
# def checkpoint_filter_fn(state_dict, model):
#     """ 转换patch embedding权重格式 """
#     out_dict = {}
#     if 'model' in state_dict:
#         state_dict = state_dict['model']
#     for k, v in state_dict.items():
#         if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
#             O, I, H, W = model.patch_embed.proj.weight.shape
#             v = v.reshape(O, -1, H, W)
#         elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
#             v = resize_pos_embed(
#                 v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
#         out_dict[k] = v
#     return out_dict
#
#
# def _create_vision_transformer(pretrained=False, default_cfg=None, **kwargs):
#     """ 创建VisionTransformer模型 """
#     if kwargs.get('features_only', None):
#         raise RuntimeError('Vision Transformer不支持features_only模式')
#
#     model = VisionTransformer(**kwargs)
#
#     if pretrained:
#         if isinstance(pretrained, str):
#             if 'npz' in pretrained:
#                 model.load_pretrained(pretrained, prefix='')
#             else:
#                 try:
#                     checkpoint = torch.load(pretrained, map_location="cpu", weights_only=False)
#                     missing_keys, unexpected_keys = model.load_state_dict(checkpoint.get("model", checkpoint),
#                                                                           strict=False)
#                     print("缺失的权重键:", missing_keys)
#                     print("多余的权重键:", unexpected_keys)
#                     print(f'成功加载预训练模型: {pretrained}')
#                 except Exception as e:
#                     print(f"加载预训练模型失败: {e}")
#                     print("警告: MAE预训练权重未加载!")
#
#     return model
#
#
# def vit_base_patch16_224(pretrained=False, **kwargs):
#     """ 创建ViT-B/16模型 """
#     model_kwargs = dict(
#         patch_size=16, in_chans=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
#     return model

