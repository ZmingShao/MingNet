from ..unet import UNet
from ..vision_transformer import ViT


class TransUnet(UNet):
    def __init__(self, in_channels, n_classes, img_size, patch_size=32):
        super(TransUnet, self).__init__(in_channels, n_classes)
        self.bottleneck = ViT(self.base_channels * self.scale,
                              image_size=tuple(map(lambda x: x // self.scale, img_size)),
                              patch_size=patch_size // self.scale,
                              depth=16)

