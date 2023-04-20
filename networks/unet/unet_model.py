""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.num_pool = 3
        self.scale = 2 ** self.num_pool
        self.base_channels = 64

        self.inc = DoubleConv(in_channels, self.base_channels)

        self.down = nn.ModuleList([Down(self.base_channels * 2 ** i,
                                        self.base_channels * 2 ** (i + 1))
                                   for i in range(self.num_pool)])

        self.bottleneck = nn.Identity()

        self.up = nn.ModuleList([Up(self.base_channels * 2 ** i,
                                    self.base_channels * 2 ** (i - 1), bilinear=False)
                                 for i in range(self.num_pool, 0, -1)])

        self.outc = OutConv(self.base_channels, n_classes)

    def forward(self, x):
        x = self.inc(x)
        feat = []
        for d in self.down:
            feat.append(x)
            x = d(x)
        x = self.bottleneck(x)
        for i, u in enumerate(self.up):
            x = u(x, feat[::-1][i])
        x = self.outc(x)
        return x
