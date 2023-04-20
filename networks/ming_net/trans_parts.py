from ..vit_pytorch.vit_for_small_dataset import SPT, Transformer as SmallDatasetTransformer
from ..trans_unet.trans_parts import ViT


class SmallDatasetViT(ViT):
    def __init__(self, channels, image_size, patch_size=4, dim=1024, depth=6, heads=16, mlp_dim=2048,
                 dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__(channels, image_size, patch_size, dim, depth, heads, mlp_dim, dim_head, dropout, emb_dropout)

        self.to_patch_embedding = SPT(dim=dim, patch_size=patch_size, channels=channels)

        self.transformer = SmallDatasetTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)
