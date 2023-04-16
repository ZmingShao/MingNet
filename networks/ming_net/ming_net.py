import torch

from ..unet.unet_parts import *
from .trans_part import SmallDatasetViT


class MingNet(nn.Module):
    def __init__(self, n_channels, n_classes, img_size, patch_size=32, bilinear=False):
        super(MingNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512 // factor))

        self.transformer = SmallDatasetViT(512 // factor,
                                           image_size=tuple(map(lambda x: x // 8, img_size)),
                                           patch_size=patch_size // 8)

        self.up3 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up1 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        x = self.transformer(x)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        return logits
