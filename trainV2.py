import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
# from tqdm.notebook import tqdm # USED FORNOTEBOOK

from eval import eval_net
from unet import UNet
from torchsummary import summary


from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BratDataSet, BratDataSetWithStacking
from torch.utils.data import DataLoader, random_split
import json


import config as cfg


dir_checkpoint = './checkpoints/'


def train_net(net, device, epochs=5, batch_size=1, lr=0.001, val_percent=0.1, save_cp=True, type='stack'):

    # Get file list
    if type == 'stack':
        fileList = ''
        with open('./stackTrain.json', 'r') as json_file:
            fileList = json.load(json_file)

        dataset = BratDataSetWithStacking(
            fileList=fileList, root=cfg.TRAIN_DATA, convert_label=True)
    else:
        fileList = ''
        with open(os.path.join(cfg.ROOT_DATA, 'train.json'), 'r') as json_file:
            fileList = json.load(json_file)

        dataset = BratDataSet(fileList=fileList, root=cfg.TRAIN_DATA)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(train)}
        Validation size: {len(val)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr,
    #                           weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, true_masks = batch['image'], batch['mask']

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        if value != None and value.data != None:
                            writer.add_histogram(
                                'weights/' + tag, value.data.cpu().numpy(), global_step)
                        if value != None and value.grad != None:
                            writer.add_histogram(
                                'grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar(
                        'learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info(
                            'Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info(
                            'Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    # writer.add_images('images', imgs, global_step)
                    # if net.n_classes == 1:
                    #     writer.add_images(
                    #         'masks/true', true_masks, global_step)
                    #     writer.add_images(
                    #         'masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train Unet Model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

    parser.add_argument('--epochs', '-e', type=int,
                        help='train epoch', default=5)
    parser.add_argument('--batch', '-b', type=int,
                        help='batch size', default=5)

    parser.add_argument('--type', '-t', help='model type', default='stack')

    parser.add_argument('--mask-threshold', '-th', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    model_checkpoint = args.model
    model_type = args.type

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')

    n_channels = 1
    n_classes = 4
    bilinear = False

    if model_type == 'stack':
        n_channels = 4
    else:
        n_channels = 1

    net = UNet(n_channels, n_classes, bilinear)

    logging.info(f'Network:\n'
                 f'\t model type {model_type}\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    logging.info('------------- Net Summary --------------')

    # summary(net, ( 4, 240, 240))

    logging.info('----------------------------------------')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch,
                  lr=0.00001,
                  device=device,
                  val_percent=20 / 100,
                  type=model_type)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

# Load Model
# net.load_state_dict(
#     torch.load(args.load, map_location=device)
# )
# logging.info(f'Model loaded from {args.load}')
