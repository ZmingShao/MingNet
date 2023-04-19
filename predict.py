import argparse
import logging
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

from utils.data_loading import CTCDataset
from utils.utils import DATA_SET, select_model

os.environ['NUMEXPR_MAX_THREADS'] = '16'

ds_name, radius = DATA_SET[3]
dir_img = Path.cwd() / ('data/train/' + ds_name + '/01')
dir_seg = Path.cwd() / ('data/train/' + ds_name + '/01_ST/SEG')
dir_track = Path.cwd() / ('data/train/' + ds_name + '/01_GT/TRA')
dir_results = Path.cwd() / ('results/' + ds_name)


def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()
    # img = torch.from_numpy(CTCDataset.preprocess(full_img, flag='image', scale=scale, patch_size=patch_size))
    img = torch.from_numpy(full_img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if isinstance(output, (tuple, list)):
            output = output[0].cpu()
        else:
            output = output.cpu()
        # output = F.interpolate(output, full_img.shape[:2], mode='bilinear')
        mask = output.argmax(dim=1) if net.n_classes > 1 else torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--input', '-i', metavar='INPUT', type=str, help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', type=str, help='Filenames of output images')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--channels', type=int, default=1, help='Number of channels; channels=3 for RGB images')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--net-name', type=str,
                        default='unet', help='select one model')
    parser.add_argument('--patch-size', '-p', type=int, default=32, help='Patch size for ViT')

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

    in_filename = args.input
    out_filename = get_output_filenames(args)

    logging.info(f'\nPredicting image {dir_img / in_filename} ...')
    img = tifffile.imread(dir_img / in_filename)
    mask = tifffile.imread(dir_seg / ('man_seg' + in_filename[1:]))
    img = CTCDataset.preprocess(img, flag='image', scale=args.scale, patch_size=args.patch_size)
    mask = CTCDataset.preprocess(mask, flag='mask', scale=args.scale, patch_size=args.patch_size)

    args.img_size = img.shape[-2:]
    net = select_model(args)

    dir_pth = dir_results / f'segment_{args.net_name}_e{args.epochs}' \
                            f'_bs{args.batch_size}_p{args.patch_size}_lr{args.lr}' \
                            f'_s{args.scale:.1f}_amp{int(args.amp)}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {dir_pth / args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(dir_pth / args.model, map_location=device)
    mask_values = [int(v / (args.classes - 1) * 255) for v in range(args.classes)] if args.classes > 1 else [255]
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    mask_pred = predict_img(net=net,
                            full_img=img,
                            out_threshold=args.mask_threshold,
                            device=device)
    # IMAGE
    img = np.uint8(255 * img.squeeze())

    # SEG
    inter = 2 * (mask_pred * mask).sum(axis=(-1, -2))
    sets_sum = mask_pred.sum(axis=(-1, -2)) + mask.sum(axis=(-1, -2))
    sets_sum = np.where(sets_sum == 0, inter, sets_sum)
    epsilon = 1e-6
    dice = (inter + epsilon) / (sets_sum + epsilon)
    logging.info(f'Dice score of segmentation: {dice:.3f}')

    mask_pred = mask_to_image(mask_pred, mask_values)
    mask = mask_to_image(mask, mask_values)
    result = np.hstack((mask, img, mask_pred))

    if not args.no_save:
        cv2.imwrite(out_filename, result)
        logging.info(f'Mask saved to {out_filename}')

    if args.viz:
        logging.info(f'Visualizing results for image {in_filename}, close to continue...')
        plt.imshow(result)
        plt.xticks([]), plt.yticks([])
        plt.show()
