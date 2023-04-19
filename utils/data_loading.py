import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import tifffile
import cv2


def load_image(filename: Path):
    ext = filename.suffix
    filename = str(filename)
    if ext == '.npy':
        return np.load(filename)
    elif ext in ['.pt', '.pth']:
        return torch.load(filename).numpy()
    elif ext == '.tif':
        return tifffile.imread(filename)
    else:
        return cv2.imread(filename)


class CTCDataset(Dataset):
    def __init__(self, ds_dir, scale=1.0, n_classes=2, patch_size=32):
        self.images_dir = Path(ds_dir) / 'images'
        self.masks_dir = Path(ds_dir) / 'masks'
        self.n_classes = n_classes
        self.scale = scale
        self.patch_size = patch_size

        self.ids = [file.name for file in self.images_dir.glob('*.tif')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self)} examples')

    def __len__(self):
        return len(self.ids)

    def image_size(self):
        img_path = list(self.images_dir.glob('*.tif'))[0]
        img = load_image(img_path)
        scaled_size = tuple(map(lambda x: int(x * self.scale), img.shape[:2]))
        padded_size = tuple(s - s % self.patch_size for s in scaled_size)
        return padded_size

    @staticmethod
    def preprocess(img: np.ndarray, flag: str, scale: float = 1.0, patch_size=32):
        new_shape = tuple(map(lambda x: int(x * scale), img.shape[:2]))
        img = cv2.resize(img, new_shape[::-1],
                         interpolation=cv2.INTER_NEAREST if flag != 'image' else cv2.INTER_CUBIC)
        # pad_y, pad_x = map(lambda x: patch_size - x % patch_size, new_shape)
        # img = np.pad(img, pad_width=((pad_y // 2, pad_y - pad_y // 2),
        #                              (pad_x // 2, pad_x - pad_x // 2)))
        diff_y, diff_x = map(lambda x: x % patch_size, new_shape)
        img = img[diff_y // 2: new_shape[0] - (diff_y - diff_y // 2),
                  diff_x // 2: new_shape[1] - (diff_x - diff_x // 2)]

        if flag == 'mask':
            img = np.uint8(img > 0)
        elif flag == 'image':
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            img = img / 255.0 if (img > 1).any() else img
        else:
            logging.error(f'Unrecognized flag: {flag}, expected `image` or `mask`')

        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name))
        mask_file = list(self.masks_dir.glob(name))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        img = load_image(img_file[0])
        mask = load_image(mask_file[0])

        assert img.shape[:2] == mask.shape, \
            f'Image and mask {name} should be the same shape, ' \
            f'but are image:{img.shape} and mask:{mask.shape}'

        img = self.preprocess(img, flag='image', scale=self.scale, patch_size=self.patch_size)
        mask = self.preprocess(mask, flag='mask', scale=self.scale, patch_size=self.patch_size)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
