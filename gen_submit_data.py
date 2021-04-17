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


from unet import UNet
from utils.dataset import BratDataSet, BratDataSetWithStacking

LAYER_SIZE = 155

OUT_DIR = './results'


def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()

    # img = torch.from_numpy(BratDataSetWithStacking.preprocess(full_img, scale_factor))

    img = torch.from_numpy(full_img)
    img = img.unsqueeze(0)

    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

    probs = probs.squeeze(0)

    full_mask = probs.squeeze().cpu().numpy()

    full_mask = full_mask > out_threshold

    seg_mask = np.zeros((240, 240))
    for i in range(5):
        seg_mask[full_mask[i]] = i

    return seg_mask


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

    parser.add_argument('--mask-threshold', '-th', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    model_checkpoint = args.model
    inputs = args.input

    logging.info(inputs)

    FILE_LIST = os.listdir(inputs)

    logging.info("Loading Model !")

    net = UNet(n_channels=4, n_classes=5, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.load_state_dict(torch.load(model_checkpoint, map_location=device))

    logging.info("Model loaded !")

    for file_name in FILE_LIST:
        if file_name.startswith('.'):
            continue
        logging.info(f'Segmenting file {file_name}')

        t1_img, t1ce_img, t2_img, flair_img = read_image(inputs, file_name)

        seg_mask = np.zeros((155, 240, 240))

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

            predict_result = predict_img(net=net, full_img=img, device=device)
            seg_mask[i] = predict_result

        write_seg_result(file_name, seg_mask)
