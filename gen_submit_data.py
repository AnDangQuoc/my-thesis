import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from numpy import newaxis
from PIL import Image
import matplotlib.pyplot as plt
import SimpleITK as sitk
from torchinfo import summary


from unet import UNet
from unetOriginal import UNet as UnetOrigin

from utils.predict import predict_img, mask_to_image
from utils.own_itk import write_itk_image

LAYER_SIZE = 155

OUT_DIR = './results'


def read_nii_file(image_path):
    img = sitk.ReadImage(image_path)
    img = sitk.GetArrayFromImage(img)
    return img


def read_image(fileDir, fileName):
    t1_img_path = os.path.join(fileDir, fileName, fileName + '_t1.nii.gz')
    t1ce_img_path = os.path.join(fileDir, fileName, fileName + '_t1ce.nii.gz')
    t2_img_path = os.path.join(fileDir, fileName, fileName + '_t2.nii.gz')
    flair_img_path = os.path.join(
        fileDir, fileName, fileName + '_flair.nii.gz')

    # Read file to numpy
    t1_img = read_nii_file(t1_img_path)
    t1ce_img = read_nii_file(t1ce_img_path)
    t2_img = read_nii_file(t2_img_path)
    flair_img = read_nii_file(flair_img_path)

    return t1_img, t1ce_img, t2_img, flair_img


def write_seg_result(fileName, result):
    converted = sitk.GetImageFromArray(result)
    sitk.WriteImage(converted, os.path.join(OUT_DIR, f'{fileName}.nii.gz'))


def get_args():
    parser = argparse.ArgumentParser(description='Test Unet Model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filenames of input images', default='./val_data',)

    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

    parser.add_argument('--type', '-t', help='model type', default='stack')

    parser.add_argument('--name', '-n', help='model name', default='origin')

    parser.add_argument('--mask-threshold', '-th', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    model_checkpoint = args.model
    model_type = args.type
    model_name = args.name

    inputs = args.input

    print(f'Input: {inputs}')

    FILE_LIST = os.listdir(inputs)

    n_channels = 1
    n_classes = 5
    bilinear = False

    if model_type == 'stack':
        n_channels = 4
    elif model_type == 'v2':
        n_channels = 4
        n_classes = 4
    else:
        n_channels = 1

    if model_name == 'origin':
        net = UnetOrigin(n_channels, n_classes, bilinear)
    else:
        net = UNet(n_channels, n_classes, bilinear)

    print('------------- Net Summary --------------')

    summary(net, input_size=(1, 4, 240, 240))

    print('----------------------------------------')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.load_state_dict(torch.load(model_checkpoint, map_location=device))

    for file_name in FILE_LIST:
        if file_name.startswith('.'):
            continue
        print(f'Segmenting file {file_name}')

        t1_img, t1ce_img, t2_img, flair_img = read_image(inputs, file_name)

        seg_mask = read_nii_file('./template/Template.nii.gz')

        for i in range(LAYER_SIZE):
            t1_layer = t1_img[i]
            t2_layer = t2_img[i]
            t1ce_layer = t1ce_img[i]
            flair_layer = flair_img[i]

            img_channel_1 = t1_layer[newaxis, :, :]
            img_channel_2 = t2_layer[newaxis, :, :]
            img_channel_3 = t1ce_layer[newaxis, :, :]
            img_channel_4 = flair_layer[newaxis, :, :]

            img = np.concatenate(
                (img_channel_1, img_channel_2, img_channel_3, img_channel_4))

            predict_result = predict_img(
                net=net, full_img=img, device=device, out_threshold=0.5)

            layer_seg_mask = predict_result
            layer_seg_mask = mask_to_image(layer_seg_mask, n_classes)

            # Convert label 3 to 4 for comparing
            if n_classes == 4:
                layer_seg_mask[layer_seg_mask == 3] = 4

            seg_mask[i] = layer_seg_mask.astype(np.int16)

        write_itk_image(seg_mask, file_name)
