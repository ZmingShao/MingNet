from ..unet.unet_parts import *
from .trans_parts import SmallDatasetViT


class MingNet(nn.Module):
    def __init__(self, in_channels, n_classes, img_size, patch_size=32):
        super(MingNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.num_pool = 4
        self.scale = 2 ** self.num_pool
        self.base_channels = 64
        assert patch_size % self.scale == 0, \
            f"Patch size should be able to downscale {self.num_pool} times"

        self.inc = DoubleConv(in_channels, self.base_channels)
        self.down = nn.ModuleList([Down(self.base_channels * 2 ** i,
                                        self.base_channels * 2 ** (i + 1))
                                   for i in range(self.num_pool)])

        self.transformer = SmallDatasetViT(self.base_channels * self.scale,
                                           image_size=tuple(map(lambda x: x // self.scale, img_size)),
                                           patch_size=patch_size // self.scale)

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
        x = self.transformer(x)
        for i, u in enumerate(self.up):
            x = u(x, feat[::-1][i])
        x = self.outc(x)
        return x
