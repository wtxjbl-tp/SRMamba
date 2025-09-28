import torch
import torch.nn as nn
import torch.nn.functional as func
from einops import rearrange
from functools import partial
from util.filter import *
from .models.vmamba import vmamba_tiny_s1l8
from .models.vmamba_up import vmamba_up_tiny_s1l8



class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        x = x.div(keep_prob) * random_tensor
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_c: int = 3, embed_dim: int = 96,
                 norm_layer: nn.Module = None, circular_padding: bool = False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.circular_padding = circular_padding
        if circular_padding:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(self.patch_size[0], 8), stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def padding(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            x = func.pad(x, (0, self.patch_size[0] - W % self.patch_size[1],
                             0, self.patch_size[1] - H % self.patch_size[0],
                             0, 0))
        return x

    # Circular padding is only used on the width of range image
    def circularpadding(self, x: torch.Tensor) -> torch.Tensor:
        x = func.pad(x, (2, 2, 0, 0), "circular")
        return x

    def forward(self, x):
        x = self.padding(x)

        if self.circular_padding:
            # Circular Padding
            x = self.circularpadding(x)

        x = self.proj(x)
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    @staticmethod
    def padding(x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        if H % 2 == 1 or W % 2 == 1:
            x = func.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        return x

    @staticmethod
    def merging(x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    def forward(self, x):
        x = self.padding(x)
        x = self.merging(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# Patch Unmerging layer
class PatchUnmerging(nn.Module):
    def __init__(self, dim: int):
        super(PatchUnmerging, self).__init__()
        self.dim = dim
        # ToDo: Use linear with norm layer?
        self.expand = nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=(1, 1))
        self.upsample = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'B H W C -> B C H W')
        x = self.expand(x.contiguous())
        x = self.upsample(x)
        # x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=1, P2=4)
        x = rearrange(x, 'B C H W -> B H W C')
        return x


# Original Patch Expanding layer used in Swin MAE
class PatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)
        # self.patch_size = patch_size

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        # x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=1, P2=4)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2, P2=2)
        x = self.norm(x)
        return x


# Original Final Patch Expanding layer used in Swin MAE
class FinalPatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm, upscale_factor=4):
        super(FinalPatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, (upscale_factor ** 2) * dim, bias=False)
        self.norm = norm_layer(dim)
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor):
        x = self.expand(x)

        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=self.upscale_factor,
                      P2=self.upscale_factor,
                      C=self.dim)
        x = self.norm(x)
        return x


class PixelShuffleHead(nn.Module):
    def __init__(self, dim: int, upscale_factor: int):
        super(PixelShuffleHead, self).__init__()
        self.dim = dim

        self.conv_expand = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim * (upscale_factor ** 2), kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True))

        # self.conv_expand = nn.Conv2d(in_channels=dim, out_channels=dim*(upscale_factor**2), kernel_size=(1, 1))
        self.upsample = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, x: torch.Tensor):
        x = self.conv_expand(x)
        x = self.upsample(x)

        return x

class SRMamba(nn.Module):
    def __init__(self, img_size=(32, 2048), target_img_size=(128, 2048), patch_size=(4, 4), in_chans: int = 1,
                 embed_dim: int = 96,
                 window_size: int = 4, depths: tuple = (2, 2, 6, 2), num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, norm_layer=nn.LayerNorm, patch_norm: bool = True,
                 pixel_shuffle: bool = False, circular_padding: bool = False, swin_v2: bool = False,
                 log_transform: bool = False,
                 patch_unmerging: bool = False):
        super().__init__()

        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path = drop_path_rate
        self.norm_layer = norm_layer
        self.img_size = img_size
        self.target_img_size = target_img_size
        self.log_transform = log_transform

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_unmerging = patch_unmerging

        if self.patch_unmerging:
            self.first_patch_expanding = PatchUnmerging(dim=embed_dim * 2 ** (len(depths) - 1))
        else:
            self.first_patch_expanding = PatchExpanding(dim=embed_dim * 2 ** (len(depths) - 1), norm_layer=norm_layer)

        self.norm_up = norm_layer(embed_dim)

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None,
                                          circular_padding=circular_padding)

        self.decoder_pred = nn.Conv2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(1, 1), bias=False)

        self.pixel_shuffle = pixel_shuffle
        self.upscale_factor = int(
            ((target_img_size[0] * target_img_size[1]) / (img_size[0] * img_size[1])) ** 0.5) * 2 * int(
            ((patch_size[0] * patch_size[1]) // 4) ** 0.5)

        if self.pixel_shuffle:
            self.ps_head = PixelShuffleHead(dim=embed_dim, upscale_factor=self.upscale_factor)
        else:
            self.final_patch_expanding = FinalPatchExpanding(dim=embed_dim, norm_layer=norm_layer,
                                                             upscale_factor=self.upscale_factor)

        self.apply(self.init_weights)
        self.vmamba_layer = vmamba_tiny_s1l8(depths=depths,embed_dim=embed_dim)
        self.vmamba_up_layer = vmamba_up_tiny_s1l8(depths=depths,embed_dim=embed_dim*len(depths))

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, pred, target):

        loss = (pred - target).abs()
        loss = loss.mean()

        if self.log_transform:
            pixel_loss = (torch.expm1(pred) - torch.expm1(target)).abs().mean()
        else:
            pixel_loss = loss.clone()

        return loss, pixel_loss

    def forward(self, x, target, eval=False, mc_drop=False):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = rearrange(x, 'B H W C -> B C H W')
        x_save = self.vmamba_layer(x)
        x = x_save[-1]
        x_save = x_save[0:4]
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.first_patch_expanding(x)
        x = rearrange(x, 'B H W C -> B C H W')
        x = self.vmamba_up_layer(x,x_save)
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.norm_up(x)
        if self.pixel_shuffle:
            x = rearrange(x, 'B H W C -> B C H W')
            x = self.ps_head(x.contiguous())
        else:
            x = self.final_patch_expanding(x)
            x = rearrange(x, 'B H W C -> B C H W')
        x = self.decoder_pred(x.contiguous())
        if mc_drop:
            return x
        else:
            total_loss, pixel_loss = self.forward_loss(x, target)
            return x, total_loss, pixel_loss


def SRMamba_tiny(**kwargs):
    model = SRMamba(
        depths=(2,2,2,2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    #  **kwargs)
    return model

def SRMamba_small(**kwargs):
    model = SRMamba(
        depths=(2,2,9,2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    #  **kwargs)
    return model

def SRMamba_medium(**kwargs):
    model = SRMamba(
        depths=(2,2,12,2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    #  **kwargs)
    return model

def SRMamba_large(**kwargs):
    model = SRMamba(
        depths=(2,2,27,2), embed_dim=96, num_heads=(3, 6, 12, 24),
        qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    #  **kwargs)
    return model




