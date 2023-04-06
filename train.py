import argparse
import logging
import os
from pathlib import Path
import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from evaluate import evaluate
from utils.data_loading import CTCDataset
from utils.loss import MultiTaskLoss
from utils.utils import DATA_SET, select_model

ds_name, radius = DATA_SET[3]
dir_ds = Path('./data/train/' + ds_name)
dir_checkpoint = Path('./checkpoints/' + ds_name)


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_size: float = 224,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        n_classes: int = 2,
        net_name: str = 'unet',
        mtl_weight: float = 0.5
):
    # 1. Create dataset
    try:
        dataset = CTCDataset(dir_ds, img_size, n_classes, radius=radius)
    except (AssertionError, RuntimeError, IndexError) as e:
        logging.error(e)
        exit(-1)

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
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_size=img_size, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images size:     {img_size}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    det_loss_fn = MultiTaskLoss(model.n_classes, task='DET')
    seg_loss_fn = MultiTaskLoss(model.n_classes, task='SEG')
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, mask_det, mask_seg = batch['image'], batch['det_mask'], batch['seg_mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_det = mask_det.to(device=device, dtype=torch.long)
                mask_seg = mask_seg.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    pred_det, pred_seg = model(images)
                    # mask_seg, mask_det = true_masks[:, 0, ...], true_masks[:, 1, ...]
                    # if model.n_classes == 1:
                    #     loss_det = criterion(pred_det.squeeze(1), mask_det.float())
                    #     loss_seg = 0.5 * criterion(pred_seg.squeeze(1), mask_seg.float()) + 0.5 * dice_loss(
                    #         F.sigmoid(pred_seg.squeeze(1)), mask_seg.float(), multiclass=False)
                    # else:
                    #     loss_det = criterion(pred_det, mask_det)
                    #     loss_seg = 0.5 * criterion(pred_seg, mask_seg) + 0.5 * dice_loss(
                    #         F.softmax(pred_seg, dim=1).float(),
                    #         F.one_hot(mask_seg, model.n_classes).permute(0, 3, 1, 2).float(),
                    #         multiclass=True
                    #     )
                    loss_det = det_loss_fn(pred_det, mask_det)
                    loss_seg = seg_loss_fn(pred_seg, mask_seg)
                    loss = (1 - mtl_weight) * loss_det + mtl_weight * loss_seg

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if mtl_weight == 0.0:
                    continue

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

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true_seg': wandb.Image(mask_seg[0].float().cpu()),
                                    'true_det': wandb.Image(mask_det[0].float().cpu()),
                                    'pred_seg': wandb.Image(pred_seg.argmax(dim=1)[0].float().cpu()),
                                    'pred_det': wandb.Image(pred_det.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            dir_pth = dir_checkpoint / f'{net_name}_w{mtl_weight:.1f}_e{epochs}_bs{batch_size}' \
                                       f'_lr{learning_rate}_sz{img_size}_amp{int(amp)}'
            Path(dir_pth).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_pth / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--img-size', '-s', type=int, default=224, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--channels', type=int, default=1, help='Number of channels; channels=3 for RGB images')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--net-name', type=str, default='unet', help='select one model')
    parser.add_argument('--mtl-weight', type=float, default=0.5,
                        help='Weight of multi-task loss: w * SEG + (1-w) * DET')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = select_model(args)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 # f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
                 )

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
    # else:
    #     model.load_from(weights=np.load(config_vit.pretrained_path))
    #     logging.info(f'Model loaded from {config_vit.pretrained_path}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_size=args.img_size,
            val_percent=args.val / 100,
            amp=args.amp,
            n_classes=args.classes,
            net_name=args.net_name,
            mtl_weight=args.mtl_weight
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        # torch.cuda.empty_cache()
        # model.use_checkpointing()
        # train_model(
        #     model=model,
        #     epochs=args.epochs,
        #     batch_size=args.batch_size,
        #     learning_rate=args.lr,
        #     device=device,
        #     img_size=args.img_size,
        #     val_percent=args.val / 100,
        #     amp=args.amp
        # )
