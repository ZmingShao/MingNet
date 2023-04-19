import re
import Augmentor
from pathlib import Path
from shutil import copyfile

from utils.utils import DATA_SET


def gen_ds(ds_dir: Path):
    images_dir = ds_dir / 'images'
    masks_dir = ds_dir / 'masks'
    if images_dir.exists() and masks_dir.exists():
        print(f'Dataset {ds_dir.name} already generated!')
        return
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    print(f'Start generation of dataset {ds_dir.name}')
    for idx in ['01', '02']:
        for img_path in (ds_dir / idx).glob('*.tif'):
            new_img_path = images_dir / img_path.name
            copyfile(img_path, new_img_path)
            new_img_path.rename(images_dir / (img_path.stem.replace('t', f'{idx}_') + img_path.suffix))
        for mask_path in (ds_dir / f'{idx}_ST' / 'SEG').glob('*.tif'):
            new_mask_path = masks_dir / mask_path.name
            copyfile(mask_path, new_mask_path)
            new_mask_path.rename(masks_dir / (mask_path.stem.replace('man_seg', f'{idx}_') + mask_path.suffix))
    print(f'Generated completely')
    print('-------------------------')


def data_augment(ds_dir: Path):
    images_dir = ds_dir / 'images'
    masks_dir = ds_dir / 'masks'
    assert images_dir.exists() and masks_dir.exists(), f'Dataset {ds_dir.name} not generated yet!'

    augmented_dir = ds_dir / 'augmented'
    if augmented_dir.exists():
        print(f'Dataset {ds_dir.name} already augmented!')
        return
    augmented_images_dir = augmented_dir / 'images'
    augmented_masks_dir = augmented_dir / 'masks'
    augmented_images_dir.mkdir(exist_ok=True, parents=True)
    augmented_masks_dir.mkdir(exist_ok=True, parents=True)
    print(f'Start augmentation of dataset {ds_dir.name}')

    p = Augmentor.Pipeline(images_dir, output_directory=str(augmented_dir))
    p.ground_truth(masks_dir)
    # p.flip_left_right(probability=0.5)
    for _ in range(5):
        p.crop_by_size(probability=1, width=256, height=256, centre=False)
        p.process()

    for path in augmented_dir.glob('*.tif'):
        if 'groundtruth_(1)_images' in path.name:
            path.rename(augmented_masks_dir / path.name.split('_')[-1])
        if 'images_original' in path.name:
            path.rename(augmented_images_dir / path.name.split('_')[-1])

    print(f'Augmented completely')
    print('-------------------------')


if __name__ == '__main__':
    dataset_dir = Path.cwd() / 'data' / 'train' / DATA_SET[3]
    gen_ds(dataset_dir)
    data_augment(dataset_dir)
