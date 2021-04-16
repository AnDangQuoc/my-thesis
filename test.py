import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BratDataSet, BratDataSetWithStacking
from torch.utils.data import DataLoader, random_split


import config as cfg


def get_args():
    parser = argparse.ArgumentParser(description='Test Unet Model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

    parser.add_argument('--type', '-t', help='model type', default='stack')

    parser.add_argument('--mask-threshold', '-th', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    model_checkpoint = args.model
    model_type = args.type

    mask_threshold = args.mask_threshold

    if model_type == 'stack':
        net = UNet(n_channels=4, n_classes=5)
        dataset = BratDataSetWithStacking(
            fileList=fileList, root=cfg.VALID_DATA)
    else:
        raise Exception('Model type not supported')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using Device: {device}')

    # Load Model Checkpoint
    net.to(device=device)
    net.load_state_dict(torch.load(model_checkpoint), map_location=device)

    logging.info('Model loaded')

    # Load Test Data
    val_loader = DataLoader(dataset, batch_size=5, shuffle=False,
                            num_workers=8, pin_memory=True, drop_last=True)

    val_score = eval_net(net, val_loader, device)

    if net.n_classes > 1:
        logging.info('Validation cross entropy: {}'.format(val_score))
    else:
        logging.info('Validation Dice Coeff: {}'.format(val_score))
