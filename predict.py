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


from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BratDataSet, BratDataSetWithStacking


COLOR = [0, 30, 60, 90, 120, 150]


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


def mask_to_image(mask):
    img = np.zeros((240, 240))
    for i in range(5):
        result = Image.fromarray((mask[i] * 255).astype(np.uint8))
        img[mask[i]] = COLOR[i]
        # result.save(f'img_{i}.png')
    return img


def show_result(img_channel_1, img_channel_2, img_channel_3, img_channel_4, result, segment):
    fig = plt.figure('Segmentation', figsize=(11, 8))

    rows = 2
    columns = 3

    fig.add_subplot(rows, columns, 1)

    plt.imshow(img_channel_1[0], cmap='gray')
    plt.title('T1 Image')

    fig.add_subplot(rows, columns, 2)

    plt.imshow(img_channel_2[0], cmap='gray')
    plt.title('T2 Image')

    fig.add_subplot(rows, columns, 3)

    plt.imshow(img_channel_3[0], cmap='gray')
    plt.title('T1ce Image')

    fig.add_subplot(rows, columns, 4)

    plt.imshow(img_channel_4[0], cmap='gray')
    plt.title('Flair Image')

    fig.add_subplot(rows, columns, 5)

    plt.imshow(result, cmap='gray')
    plt.title('Predict')

    fig.add_subplot(rows, columns, 6)

    plt.imshow(segment*30, cmap='gray')
    plt.title('Ground Truth')

    plt.subplots_adjust(hspace=0.2, top=0.9, bottom=0.1)
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Test Unet Model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filenames of input images', required=True)

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

    file_name = args.input

    model_checkpoint = args.model

    net = UNet(n_channels=4, n_classes=5, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.load_state_dict(torch.load(model_checkpoint, map_location=device))

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

    img_channel_1 = img_channel_1[newaxis, :, :]
    img_channel_2 = img_channel_2[newaxis, :, :]
    img_channel_3 = img_channel_3[newaxis, :, :]
    img_channel_4 = img_channel_4[newaxis, :, :]

    segment = np.load(os.path.join(root, f'{file_name}_004.npy'))
    img = np.concatenate(
        (img_channel_1, img_channel_2, img_channel_3, img_channel_4))

    mask = predict_img(net=net, full_img=img, device=device)

    result = mask_to_image(mask)

    show_result(img_channel_1, img_channel_2, img_channel_3,
                img_channel_4, result, segment)
