import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from ..vit_pytorch.vit import pair, Transformer
from ..vit_pytorch.vit_for_small_dataset import SPT, Transformer as SmallDatasetTransformer


class ViT(nn.Module):
    def __init__(self, channels, image_size, patch_size=4, dim=1024, depth=6, heads=16, mlp_dim=2048,
                 dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        self.num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
        self.patch_dim = channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.decoder = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_height, p2=self.patch_width,
                      h=self.image_height // self.patch_height, w=self.image_width // self.patch_width),
            nn.Conv2d(dim // (self.patch_height * self.patch_width), channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_latent(x)
        return self.decoder(x)


class SmallDatasetViT(ViT):
    def __init__(self, channels, image_size, patch_size=4, dim=1024, depth=6, heads=16, mlp_dim=2048,
                 dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__(channels, image_size, patch_size, dim, depth, heads, mlp_dim, dim_head, dropout, emb_dropout)

        self.to_patch_embedding = SPT(dim=dim, patch_size=patch_size, channels=channels)

        self.transformer = SmallDatasetTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)
