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


from unetOriginal import UNet
from CAUNet import UNet as CAUNet
from MCAUNet import UNet as CAUMaxUNet
from utils.dataset import BratDataSet, BratDataSetWithStacking


COLOR = [0, 30, 60, 90, 120, 150]

RED = [0, 44, 252, 0, 252]
GREEN = [0, 122, 246, 0, 246]
BLUE = [0, 230, 53, 0, 53]

LIGHT_BLUE = [44, 112, 230]
LIGHT_YELLOW = [252, 246, 53]
LIGHT_RED = [252, 246, 53]


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
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

        # probs = probs.squeeze(0)

        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(full_img.size[1]),
        #         transforms.ToTensor()
        #     ]
        # )

        # probs = tf(probs.cpu())
        # full_mask = probs.squeeze().cpu().numpy()

    probs = probs.squeeze(0)

    full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def mask_to_image(mask, isGroundTruth=False):
    img = np.zeros((240, 240), dtype=np.uint8)

    if isGroundTruth:
        for i in range(4):
            newMask = mask == i
            img[newMask] = COLOR[i]
            # img[0][newMask] = RED[i]
            # img[1][newMask] = GREEN[i]
            # img[2][newMask] = BLUE[i]
    else:
        for i in range(4):
            img[mask[i]] = COLOR[i]
            # img[0][mask[i]] = RED[i]
            # img[1][mask[i]] = GREEN[i]
            # img[2][mask[i]] = BLUE[i]
            # result = Image.fromarray((mask[i] * 255).astype(np.uint8))
            # result.save(f'img_{i}.png')

    return img


def show_result(file_name, img_channel_1, img_channel_2, img_channel_3, img_channel_4, result_origin, result_CA, result_Max, segment):
    fig = plt.figure(f'Segmentation {file_name}', figsize=(11, 8))

    rows = 1
    columns = 4

    # fig.add_subplot(rows, columns, 1)

    # plt.imshow(img_channel_1[0], cmap='gray')
    # plt.title('T1 Image')
    # plt.axis('off')

    # fig.add_subplot(rows, columns, 2)

    # plt.imshow(img_channel_2[0], cmap='gray')
    # plt.title('T2 Image')
    # plt.axis('off')

    # fig.add_subplot(rows, columns, 3)

    # plt.imshow(img_channel_3[0], cmap='gray')
    # plt.title('T1ce Image')
    # plt.axis('off')

    # fig.add_subplot(rows, columns, 4)

    # plt.imshow(img_channel_4[0], cmap='gray')
    # plt.title('Flair Image')
    # plt.axis('off')

    fig.add_subplot(rows, columns, 1)

    plt.imshow(result_origin)
    plt.title('UNet')
    plt.axis('off')

    fig.add_subplot(rows, columns, 2)

    plt.imshow(result_CA)
    plt.title('CA-UNet')
    plt.axis('off')

    fig.add_subplot(rows, columns, 3)

    plt.imshow(result_Max)
    plt.title('MCA-UNet')
    plt.axis('off')

    fig.add_subplot(rows, columns, 4)

    plt.imshow(segment*30)
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplots_adjust(hspace=0.5)

    # Save result
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Test Unet Model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filenames of input images', required=True)

    parser.add_argument('--mask-threshold', '-th', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    file_name = args.input

    originNet = UNet(n_channels=4, n_classes=4, bilinear=False)
    CANet = CAUNet(n_channels=4, n_classes=4, bilinear=False)
    MaxPoolNet = CAUMaxUNet(n_channels=4, n_classes=4, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    originNet.load_state_dict(torch.load(
        'checkpoints/UNet/CP_epoch5.pth', map_location=device))

    CANet.load_state_dict(torch.load(
        'checkpoints/CAUNet/CP_epoch5.pth', map_location=device))

    MaxPoolNet.load_state_dict(torch.load(
        'checkpoints/MCAUNet/CP_epoch5.pth', map_location=device))

    logging.info("Model loaded !")

    root = './data/train'
    img_channel_1 = f'{file_name}_000.npy'
    img_channel_2 = f'{file_name}_001.npy'
    img_channel_3 = f'{file_name}_002.npy'
    img_channel_4 = f'{file_name}_003.npy'

    img_channel_1 = np.load(os.path.join(root, img_channel_1))
    img_channel_2 = np.load(os.path.join(root, img_channel_2))
    img_channel_3 = np.load(os.path.join(root, img_channel_3))
    img_channel_4 = np.load(os.path.join(root, img_channel_4))

    img_channel_1 = img_channel_1 / np.max(img_channel_1)
    img_channel_2 = img_channel_2 / np.max(img_channel_2)
    img_channel_3 = img_channel_3 / np.max(img_channel_3)
    img_channel_4 = img_channel_4 / np.max(img_channel_4)

    img_channel_1 = img_channel_1[newaxis, :, :]
    img_channel_2 = img_channel_2[newaxis, :, :]
    img_channel_3 = img_channel_3[newaxis, :, :]
    img_channel_4 = img_channel_4[newaxis, :, :]

    segment = np.load(os.path.join(root, f'{file_name}_004.npy'))
    img = np.concatenate(
        (img_channel_1, img_channel_2, img_channel_3, img_channel_4))

    mask_origin = predict_img(net=originNet, full_img=img, device=device)
    mask_CA = predict_img(net=CANet, full_img=img, device=device)
    mask_Max = predict_img(net=MaxPoolNet, full_img=img, device=device)

    result_origin = mask_to_image(mask_origin)
    result_CA = mask_to_image(mask_CA)
    result_Max = mask_to_image(mask_Max)

    show_result(file_name, img_channel_1, img_channel_2, img_channel_3,
                img_channel_4, result_origin, result_CA, result_Max, segment)
