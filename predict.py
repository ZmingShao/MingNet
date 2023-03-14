import argparse
import logging
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import tifffile
from pathlib import Path

from utils.data_loading import CTCDataset
from networks.trans_unet import VisionTransformer, CONFIGS as CONFIGS_vit
from networks.unet import UNet
from networks.swin_unet import SwinUnet, get_config as get_config_swin
from networks.unet_plus_plus import Generic_UNetPlusPlus, softmax_helper

from utils.utils import DATA_SET, det_vis

os.environ['NUMEXPR_MAX_THREADS'] = '16'

ds_name, radius = DATA_SET[3]
# dir_img = Path('./data/test/' + ds_name + '/02')
dir_img = Path('./data/train/' + ds_name + '/01')
dir_seg = Path('./data/train/' + ds_name + '/01_ST/SEG')
dir_track = Path('./data/train/' + ds_name + '/01_GT/TRA')
dir_checkpoint = Path('./checkpoints/' + ds_name + '/w1.0_e10_bs4_lr1e-05_sz512_amp1')


def predict_img(net,
                full_img,
                device,
                img_size=224,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(CTCDataset.preprocess(full_img, img_size, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        outputs = net(img)
        # DET
        output = outputs[0]
        if isinstance(output, (tuple, list)):
            output = output[0].cpu()
        else:
            output = output.cpu()
        output = F.interpolate(output, full_img.shape[:2], mode='bilinear')
        mask_pred = output.argmax(dim=1) if net.n_classes > 1 else torch.sigmoid(output) > out_threshold
        mask_pred_det = mask_pred[0].long().squeeze().numpy()
        # SEG
        output = outputs[1]
        if isinstance(output, (tuple, list)):
            output = output[0].cpu()
        else:
            output = output.cpu()
        output = F.interpolate(output, full_img.shape[:2], mode='bilinear')
        mask_pred = output.argmax(dim=1) if net.n_classes > 1 else torch.sigmoid(output) > out_threshold
        mask_pred_seg = mask_pred[0].long().squeeze().numpy()

    return mask_pred_det, mask_pred_seg


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--img-size', '-s', type=int, default=224,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--channels', type=int, default=1, help='Number of channels; channels=3 for RGB images')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    # parser.add_argument('--n-skip', type=int,
    #                     default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--net-name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return out


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    if args.net_name == 'unet':
        net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    elif args.net_name == 'trans_unet':
        args.net_name = 'R50-ViT-B_16'
        config_vit = CONFIGS_vit[args.net_name]
        config_vit.n_classes = args.classes
        config_vit.n_skip = 3
        if args.net_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / 16), int(args.img_size / 16))
        net = VisionTransformer(config_vit,
                                img_size=args.img_size,
                                n_classes=args.classes,
                                n_channels=args.channels)
    elif args.net_name == 'swin_unet':
        args.cfg = 'networks/swin_unet/swin_tiny_patch4_window7_224_lite.yaml'
        config_swin = get_config_swin(args)
        net = SwinUnet(config_swin, args.img_size, args.classes, args.channels)
    elif args.net_name == 'unet_pp':
        dropout_op = nn.Dropout2d
        norm_op = nn.InstanceNorm2d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        base_num_features, num_pool = 30, 5
        net = Generic_UNetPlusPlus(args.channels, base_num_features, args.classes, num_pool, norm_op=norm_op,
                                   norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                                   dropout_op_kwargs=dropout_op_kwargs, nonlin=net_nonlin,
                                   nonlin_kwargs=net_nonlin_kwargs, final_nonlin=lambda x: x,
                                   convolutional_pooling=True, convolutional_upsampling=True)
        net.inference_apply_nonlin = softmax_helper
    else:
        logging.error('Model not found!')
        exit(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {dir_checkpoint / args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(dir_checkpoint / args.model, map_location=device)
    # mask_values = state_dict.pop('mask_values')
    mask_values = [int(v / (args.classes - 1) * 255) for v in range(args.classes)] if args.classes > 1 else [255]
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {dir_img / filename} ...')
        # img = Image.open(filename)
        img = tifffile.imread(dir_img / filename)
        seg_mask = tifffile.imread(dir_seg / ('man_seg' + filename[1:]))
        tra_mask = tifffile.imread(dir_track / ('man_track' + filename[1:]))
        mask_true = np.stack((seg_mask, tra_mask), axis=-1)
        mask_true = CTCDataset.preprocess(mask_true, -1, is_mask=True, radius=radius)
        seg_mask, det_mask = mask_true[0, ...], mask_true[1, ...]

        det_mask_pred, seg_mask_pred = predict_img(net=net,
                                                   full_img=img,
                                                   img_size=args.img_size,
                                                   out_threshold=args.mask_threshold,
                                                   device=device)

        # DET
        det_mask_pred = mask_to_image(det_mask_pred, mask_values)
        result_pred = det_vis(img, det_mask_pred, args.classes, radius=radius)

        det_mask = mask_to_image(det_mask, mask_values)
        result_true = det_vis(img, det_mask, args.classes, radius=radius)

        result_det = np.hstack((result_true, result_pred))

        # SEG
        inter = 2 * (seg_mask_pred * seg_mask).sum(axis=(-1, -2))
        sets_sum = seg_mask_pred.sum(axis=(-1, -2)) + seg_mask.sum(axis=(-1, -2))
        sets_sum = np.where(sets_sum == 0, inter, sets_sum)
        epsilon = 1e-6
        dice = (inter + epsilon) / (sets_sum + epsilon)
        logging.info(f'Dice score of segmentation: {dice:.3f}')

        seg_mask_pred = mask_to_image(seg_mask_pred, mask_values)
        seg_mask = mask_to_image(seg_mask, mask_values)
        result_seg = np.hstack((seg_mask, seg_mask_pred))
        result_seg = cv2.cvtColor(result_seg, cv2.COLOR_GRAY2RGB)

        # RESULT
        result_arr = np.vstack((result_det, result_seg))
        result_arr = np.asarray(result_arr, dtype='u1')
        result = Image.fromarray(result_arr)

        if not args.no_save:
            out_filename = out_files[i]
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            # plot_img_and_mask(img, mask)
            plt.figure(1)
            plt.imshow(det_mask_pred)
            plt.figure(2)
            plt.imshow(result_pred)
            # plt.imshow(result)
            plt.xticks([]), plt.yticks([])
            plt.show()
