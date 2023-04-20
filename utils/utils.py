import torch.nn as nn

from networks.ming_net import MingNet
from networks.trans_unet import TransUnet
from networks.unet import UNet
from networks.unet_plus_plus import Generic_UNetPlusPlus, softmax_helper


def select_model(args):
    if args.net_name == 'ming_net':
        model = MingNet(in_channels=args.channels,
                        n_classes=args.classes,
                        img_size=args.img_size,
                        patch_size=args.patch_size)
    elif args.net_name == 'unet':
        model = UNet(in_channels=args.channels, n_classes=args.classes)
    elif args.net_name == 'trans_unet':
        model = TransUnet(in_channels=args.channels,
                          n_classes=args.classes,
                          img_size=args.img_size,
                          patch_size=args.patch_size)
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
