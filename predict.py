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
from utils.utils import DATA_SET, det_vis, select_model

os.environ['NUMEXPR_MAX_THREADS'] = '16'

ds_name, radius = DATA_SET[3]
# dir_img = Path('./data/test/' + ds_name + '/02')
dir_img = Path('./data/train/' + ds_name + '/01')
dir_seg = Path('./data/train/' + ds_name + '/01_ST/SEG')
dir_track = Path('./data/train/' + ds_name + '/01_GT/TRA')
dir_checkpoint = Path('./checkpoints/' + ds_name)


def predict_img(net,
                full_img,
                device,
                img_size=224,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(CTCDataset.preprocess(full_img, img_size, flag='image'))
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
        mask_pred_det = (mask_pred[0] > 0.5).long().squeeze().numpy()
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
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--img-size', '-s', type=int, default=224,
                        help='Scale factor for the input images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--channels', type=int, default=1, help='Number of channels; channels=3 for RGB images')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--net-name', type=str,
                        default='unet', help='select one model')
    parser.add_argument('--mtl-weight', type=float, default=0.5,
                        help='Weight of multi-task loss: w * SEG + (1-w) * DET')

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

    net = select_model(args)

    dir_pth = dir_checkpoint / f'{args.net_name}_w{args.mtl_weight:.1f}_e{args.epochs}' \
                                      f'_bs{args.batch_size}_lr{args.lr}' \
                                      f'_sz{args.img_size}_amp{int(args.amp)}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {dir_pth / args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(dir_pth / args.model, map_location=device)
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
        # mask_true = np.stack((seg_mask, tra_mask), axis=-1)
        seg_mask = CTCDataset.preprocess(seg_mask, -1, flag='seg_mask', radius=radius)
        det_mask = CTCDataset.preprocess(tra_mask, -1, flag='det_mask', radius=radius)
        # seg_mask, det_mask = mask_true[0, ...], mask_true[1, ...]

        det_mask_pred, seg_mask_pred = predict_img(net=net,
                                                   full_img=img,
                                                   img_size=args.img_size,
                                                   out_threshold=args.mask_threshold,
                                                   device=device)

        # DET
        det_mask_pred = mask_to_image(det_mask_pred, mask_values)
        result_pred = det_vis(img, det_mask_pred, mask_values, radius=radius)

        det_mask = mask_to_image(det_mask, mask_values)
        result_true = det_vis(img, det_mask, mask_values, radius=radius)

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
        result = np.vstack((result_det, result_seg))
        # result = np.asarray(result, dtype='u1')
        # result = Image.fromarray(result_arr)

        if not args.no_save:
            out_filename = out_files[i]
            # result.save(out_filename)
            cv2.imwrite(out_filename, result)
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
