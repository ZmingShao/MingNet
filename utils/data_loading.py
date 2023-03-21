import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import tifffile
import cv2
from typing import List

from utils.utils import draw_umich_gaussian as draw_gaussian


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')


def cnt_center(cnt: List[tuple]):
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    else:
        while isinstance(cnt[0], np.ndarray):
            cnt = cnt[0]
        center = cnt
    return center


class CTCDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=224, n_classes=2, radius=3):
        self.images_dir = Path(images_dir)
        self.seg_dir = Path(masks_dir['SEG'])
        self.track_dir = Path(masks_dir['TRA'])
        self.n_classes = n_classes
        # self.mask_values = [int(v / (n_classes - 1) * 255) for v in range(n_classes)] if n_classes > 1 else [255]
        self.img_size = img_size
        self.radius = radius

        self.ids = [splitext(str(file))[0] for file in listdir(images_dir) if not str(file).startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(img: np.ndarray, img_size: int, flag: str, radius=3):
        if img_size > 0:
            img = cv2.resize(img, (img_size, img_size),
                             interpolation=cv2.INTER_NEAREST if flag != 'image' else cv2.INTER_CUBIC)

        if flag == 'det_mask':
            img = 255 * np.uint8(img > 0)
            # img = img.transpose((2, 0, 1)) / 255.0
            # seg_bin, tra_bin = img[..., 0], img[..., 1]
            cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            img = np.zeros_like(img, dtype=np.float64)
            for cnt in cnts:
                center = cnt_center(cnt)
                # cv2.circle(det_bin, center, radius, 255, -1)
                draw_gaussian(img, center, radius)
            # diff = cv2.absdiff(det_bin, seg_bin)
            # cnts, hier = cv2.findContours(diff, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            # hier = np.squeeze(hier)
            # det_mask = np.zeros_like(diff)
            # seg_mask = np.zeros_like(diff)
            # for i in range(len(cnts)):
            #     first_child = hier[i, -2]
            #     if first_child != -1:
            #         next_child = hier[first_child, 0]
            #         ct_first, ct_next = cnt_center(cnts[first_child]), cnt_center(cnts[next_child])
            #         if next_child == -1:
            #             cv2.circle(det_mask, ct_first, radius, 1, -1)
            #             # cv2.drawContours(det_mask, cnts, first_child, 1, -1)
            #             cv2.drawContours(seg_mask, cnts, i, 1, -1)
            #         else:
            #             cv2.circle(det_mask, ct_first, radius, 1, -1)
            #             cv2.circle(det_mask, ct_next, radius, 1, -1)
            #             # cv2.drawContours(det_mask, cnts, first_child, 2, -1)
            #             # cv2.drawContours(det_mask, cnts, next_child, 2, -1)
            #             cv2.drawContours(seg_mask, cnts, i, 1, -1)
            #         # cv2.drawContours(det_mask, cnts, i, 50, 1)
            # img = np.stack((seg_bin, det_bin), axis=0)
            # img = np.uint8(img / 255)
        elif flag == 'seg_mask':
            img = np.uint8(img > 0)
        elif flag == 'image':
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            img = img / 255.0 if (img > 1).any() else img
        else:
            print(f'Unrecognized flag: {flag}, expected `image`,`det_mask` or `seg_mask`')

        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.tif'))
        seg_file = list(self.seg_dir.glob(name.replace('t', 'man_seg') + '.tif'))
        track_file = list(self.track_dir.glob(name.replace('t', 'man_track') + '.tif'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(seg_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {seg_file}'
        assert len(track_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {track_file}'
        img = tifffile.imread(img_file[0])
        seg_mask = tifffile.imread(seg_file[0])
        track_mask = tifffile.imread(track_file[0])
        # mask = np.stack((seg_mask, track_mask), axis=-1)

        assert img.shape[:2] == seg_mask.shape == track_mask.shape, \
            f'Image and mask {name} should be the same shape, ' \
            f'but are image:{img.shape} and mask:{seg_mask.shape}, {track_mask.shape}'

        img = self.preprocess(img, self.img_size, flag='image', radius=self.radius)
        det_mask = self.preprocess(track_mask, self.img_size, flag='det_mask', radius=self.radius)
        seg_mask = self.preprocess(seg_mask, self.img_size, flag='seg_mask', radius=self.radius)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'det_mask': torch.as_tensor(det_mask.copy()).float().contiguous(),
            'seg_mask': torch.as_tensor(seg_mask.copy()).long().contiguous()
        }
        # return {'image': img, 'mask': mask}
