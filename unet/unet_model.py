""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .trans_parts import VisionTransformer
from .model_configs import get_vit_config


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, img_size=224, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512 // factor))
        # self.down4 = (Down(512, 1024 // factor))
        # self.up4 = (Up(1024, 512 // factor, bilinear))
        self.up3 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up1 = (Up(128, 64, bilinear))
        self.outc_det = (OutConv(64, n_classes))
        self.outc_seg = (OutConv(64, n_classes))

        vit_config = get_vit_config()
        vit_config.channels['in'] = 512 // factor
        vit_config.channels['out'] = 512 // factor
        self.bottleneck = VisionTransformer(vit_config, img_size)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        # x = self.down4(x4)
        x = self.bottleneck(x)
        # x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        det, seg = self.outc_det(x), self.outc_seg(x)
        return det, seg

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc_det = torch.utils.checkpoint(self.outc_det)
        self.outc_seg = torch.utils.checkpoint(self.outc_seg)
