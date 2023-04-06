import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from ..vit_pytorch.vit_for_small_dataset import pair, SPT, Transformer


class SmallDatasetViT(nn.Module):
    def __init__(self, channels, image_size, patch_size=8, dim=1024, depth=6, heads=16, mlp_dim=2048,
                 dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.to_patch_embedding = SPT(dim=dim, patch_size=patch_size, channels=channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.decoder = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width,
                      h=image_height // patch_height, w=image_width // patch_width),
            nn.Conv2d(dim // (patch_height * patch_width), channels, kernel_size=3, padding=1),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_latent(x)
        return self.decoder(x)
