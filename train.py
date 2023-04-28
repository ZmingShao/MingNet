import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import csv

from evaluate import evaluate
from utils.data_loading import CTCDataset
from utils.utils import DATA_SET, select_model
from utils.loss import LossFn

ds_name = DATA_SET[8]
dir_ds = Path.cwd() / ('data/train/' + ds_name)
dir_results = Path.cwd() / ('results/' + ds_name)


def train_model(
        model,
        dataset,
        device,
        n_classes: int = 2,
        epochs: int = 5,
        batch_size: int = 1,
        patch_size: int = 32,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        img_scale: float = 0.5,
        img_size: int = 512,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        net_name: str = 'unet',
        save_results: bool = False,
):
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, img_size=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Patch size:      {patch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Images size:     {img_size}
        Mixed Precision: {amp}
    ''')

    # (path to save results)
    if save_results:
        dir_pth = dir_results / f'segment_{net_name}_e{epochs}_bs{batch_size}_p{patch_size}' \
                                f'_lr{learning_rate}_s{img_scale:.1f}_amp{int(amp)}'
        Path(dir_pth).mkdir(parents=True, exist_ok=True)
        csv_file = (dir_pth / 'result.csv').open('w')
        csv_wrt = csv.writer(csv_file)
        csv_wrt.writerow(['Epoch', 'Loss', 'Dice'])

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_fn = LossFn(n_classes=n_classes)
    global_step = 0

    # 5. Begin training
    val_score_list = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, val_score = [], 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = loss_fn(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                # epoch_loss += loss.item()
                epoch_loss.append(loss.item())
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info(f'Validation Dice score: {val_score}')
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        val_score = val_score.item()
        if save_results:
            csv_wrt.writerow([epoch, epoch_loss, val_score])
            if len(val_score_list):
                if val_score >= val_score_list[-1]:
                    torch.save(model.state_dict(), str(dir_pth / f'best.pth'))
        val_score_list.append(val_score)

    best_score = max(val_score_list)
    best_epoch = val_score_list.index(best_score) + 1
    if save_results:
        csv_file.close()
        torch.save(model.state_dict(), str(dir_pth / f'last.pth'))
        logging.info(f'Results saved in {dir_pth}')
    logging.info(f'Best dice score: {best_score}, validated in epoch {best_epoch}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--channels', type=int, default=1, help='Number of channels; channels=3 for RGB images')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--net-name', type=str, default='unet', help='select one model')
    parser.add_argument('--save-results', action='store_true', default=False, help='Save the training pth and csv')
    parser.add_argument('--patch-size', '-p', type=int, default=32, help='Patch size for ViT')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 1. Create dataset
    try:
        dataset = CTCDataset(dir_ds, args.scale, args.classes, args.patch_size)
    except (AssertionError, RuntimeError, IndexError) as e:
        logging.error(e)
        exit(-1)

    args.img_size = dataset.image_size()
    model = select_model(args)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 )

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(model=model,
                    dataset=dataset,
                    device=device,
                    n_classes=args.classes,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    patch_size=args.patch_size,
                    learning_rate=args.lr,
                    val_percent=args.val / 100,
                    img_scale=args.scale,
                    img_size=args.img_size,
                    amp=args.amp,
                    net_name=args.net_name,
                    save_results=args.save_results)
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
