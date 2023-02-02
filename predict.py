import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import tifffile
from pathlib import Path

from utils.data_loading import CTCDataset
from unet import UNet
from utils.utils import plot_img_and_mask, DATA_SET, det_vis

os.environ['NUMEXPR_MAX_THREADS'] = '16'

ds_name = DATA_SET[3]
# dir_img = Path('./data/test/' + ds_name + '/02')
dir_img = Path('./data/train/' + ds_name + '/01')
dir_seg = Path('./data/train/' + ds_name + '/01_ST/SEG')
dir_track = Path('./data/train/' + ds_name + '/01_GT/TRA')
dir_checkpoint = Path('./checkpoints/' + ds_name + '/w0.3_e10_bs6_lr1e-05_sz512_amp1')


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
        output = net(img)[0].cpu()
        output = F.interpolate(output, full_img.shape[:2], mode='bilinear')

        if net.n_classes > 1:
            mask_pred = output.argmax(dim=1)
        else:
            mask_pred = torch.sigmoid(output) > out_threshold

    return mask_pred[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
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

    net = UNet(n_channels=args.channels, n_classes=args.classes, img_size=args.img_size, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
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
        mask_true = CTCDataset.preprocess(mask_true, -1, is_mask=True)
        seg_mask, det_mask = mask_true[0, ...], mask_true[1, ...]

        mask = predict_img(net=net,
                           full_img=img,
                           img_size=args.img_size,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            det_mask_pred = mask_to_image(mask, mask_values)
            result_pred = det_vis(img, det_mask_pred, args.classes)

            det_mask = mask_to_image(det_mask, mask_values)
            result_true = det_vis(img, det_mask, args.classes)

            result = Image.fromarray(np.hstack((result_true, result_pred)))
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
