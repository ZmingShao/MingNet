import torch.nn as nn

from networks.ming_net import MingNet
from networks.trans_unet import VisionTransformer, CONFIGS as CONFIGS_vit
from networks.unet import UNet
from networks.swin_unet import SwinUnet, get_config as get_config_swin
from networks.unet_plus_plus import Generic_UNetPlusPlus, softmax_helper


def select_model(args):
    if args.net_name == 'ming_net':
        model = MingNet(in_channels=args.channels,
                        n_classes=args.classes,
                        img_size=args.img_size,
                        patch_size=args.patch_size)
    elif args.net_name == 'unet':
        model = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    elif args.net_name == 'trans_unet':
        net_name = 'R50-ViT-B_16'
        config_vit = CONFIGS_vit[net_name]
        config_vit.n_classes = args.classes
        config_vit.n_skip = 3
        if net_name.find('R50') != -1:
            config_vit.patches.grid = tuple(map(lambda x: x // 16, args.img_size))
        model = VisionTransformer(config_vit,
                                  img_size=args.img_size,
                                  n_classes=args.classes,
                                  n_channels=args.channels)
    elif args.net_name == 'swin_unet':
        args.cfg = 'networks/swin_unet/swin_tiny_patch4_window7_224_lite.yaml'
        config_swin = get_config_swin(args)
        model = SwinUnet(config_swin, args.img_size, args.patch_size, args.classes, args.channels)
    elif args.net_name == 'unet_pp':
        dropout_op = nn.Dropout2d
        norm_op = nn.InstanceNorm2d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        base_num_features, num_pool = 32, 5
        model = Generic_UNetPlusPlus(args.channels, base_num_features, args.classes, num_pool, norm_op=norm_op,
                                     norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                                     dropout_op_kwargs=dropout_op_kwargs, nonlin=net_nonlin,
                                     nonlin_kwargs=net_nonlin_kwargs, final_nonlin=lambda x: x,
                                     convolutional_pooling=True, convolutional_upsampling=True)
        model.inference_apply_nonlin = softmax_helper
    else:
        model = None
        print('Model not found!')

    return model


DATA_SET = {0: "Fluo-N2DH-SIM+",
            1: "Fluo-C2DL-MSC",
            2: "Fluo-N2DH-GOWT1",
            3: "PhC-C2DL-PSC",
            4: "BF-C2DL-HSC",
            5: "Fluo-N2DL-HeLa",
            6: "BF-C2DL-MuSC",
            7: "DIC-C2DH-HeLa",
            8: "PhC-C2DH-U373"}
