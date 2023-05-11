from networks.ming_net import MingNet
from networks.trans_unet import TransUnet
from networks.unet import UNet


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
            8: "PhC-C2DH-U373",
            9: "Fluo-C2DL-Huh7"}
