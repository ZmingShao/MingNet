import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn

from networks.ming_net import MingNet
from networks.trans_unet import VisionTransformer, CONFIGS as CONFIGS_vit
from networks.unet import UNet
from networks.swin_unet import SwinUnet, get_config as get_config_swin
from networks.unet_plus_plus import Generic_UNetPlusPlus, softmax_helper


def det_vis(img, mask, mask_values, radius=3):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = [(255, 0, 0), (0, 0, 255)]
    for i, v in enumerate(mask_values[1:]):
        cnts, _ = cv2.findContours(np.uint8(mask == v), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            # area_true = np.pi * radius ** 2
            # if cv2.contourArea(cnt) < area_true * 0.3 and radius > 5:
            #     continue
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            else:
                while isinstance(cnt[0], np.ndarray):
                    cnt = cnt[0]
                center = cnt
            cv2.circle(img, center, 1, colors[i], -1)
    return img


# def plot_img_and_mask(img, mask):
#     classes = mask.max() + 1
#     fig, ax = plt.subplots(1, classes)
#     ax[0].set_title('Input image')
#     ax[0].imshow(img)
#     ax[0].set_xticks([]), ax[0].set_yticks([])
#     for i in range(1, classes):
#         ax[i].set_title(f'Mask (class {i})')
#         ax[i].imshow(mask == i)
#         ax[i].set_xticks([]), ax[i].set_yticks([])
#     plt.show()

def select_model(args):
    if args.net_name == 'ming_net':
        model = MingNet(n_channels=args.channels,
                        n_classes=args.classes,
                        img_size=args.img_size,
                        patch_size=args.patch_size,
                        bilinear=args.bilinear)
    elif args.net_name == 'unet':
        model = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    elif args.net_name == 'trans_unet':
        net_name = 'R50-ViT-B_16'
        config_vit = CONFIGS_vit[net_name]
        config_vit.n_classes = args.classes
        config_vit.n_skip = 3
        if net_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.scale / 16), int(args.scale / 16))
        model = VisionTransformer(config_vit,
                                  img_size=args.img_size,
                                  n_classes=args.classes,
                                  n_channels=args.channels)
    elif args.net_name == 'swin_unet':
        args.cfg = 'networks/swin_unet/swin_tiny_patch4_window7_224_lite.yaml'
        config_swin = get_config_swin(args)
        model = SwinUnet(config_swin, args.img_size, args.classes, args.channels)
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


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


DATA_SET = {0: ("Fluo-N2DH-SIM+", 10),
            1: ("Fluo-C2DL-MSC", 10),
            2: ("Fluo-N2DH-GOWT1", 20),
            3: ("PhC-C2DL-PSC", 5),
            4: ("BF-C2DL-HSC", 10),
            5: ("Fluo-N2DL-HeLa", 10),
            6: ("BF-C2DL-MuSC", 10),
            7: ("DIC-C2DH-HeLa", 10),
            8: ("PhC-C2DH-U373", 15)}
