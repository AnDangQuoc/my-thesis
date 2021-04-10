import SimpleITK as sitk
import torch
from torch.utils import data
import numpy as np
import logging
import os
from numpy import newaxis


class BratDataSet(data.Dataset):
    def __init__(self, fileList: str, root: str):
        self.fileList = fileList
        self.root = root

        logging.info(f'Creating dataset with {len(self.fileList)} examples')

    def __len__(self):
        return len(self.fileList)

    def read_image(self, fileName, root):
        img = np.load(os.path.join(root, fileName))

        return img

    def get_mask_file(self, fileName):
        tmp = fileName.split('_')
        tmp[-1] = '004.npy'

        result = ''

        for i in range(len(tmp)):
            result = result + tmp[i]
            if i < (len(tmp)-1):
                result = result + '_'
        return result

    def __getitem__(self, index):
        # Select sample
        img_ID = self.fileList[index]
        mask_ID = self.get_mask_file(img_ID)

        # Load input and groundtruth
        img = self.read_image(img_ID, self.root)
        img = img[newaxis, :, :]
        mask = self.read_image(mask_ID, self.root)
        return {
            'image':  torch.from_numpy(img).type(torch.FloatTensor),
            'mask':  torch.from_numpy(mask).type(torch.FloatTensor)
        }


class BratDataSetWithStacking(data.Dataset):
    def __init__(self, fileList: str, root: str):
        self.fileList = fileList
        self.root = root

        logging.info(f'Creating dataset with {len(self.fileList)} examples')

    def __len__(self):
        return len(self.fileList)

    def read_image(self, fileName, root):
        img = np.load(os.path.join(root, fileName))
        return img

    def read_train_image(self, img_ID, root):
        t1_img = self.read_image(img_ID+'_000.npy', self.root)
        t2_img = self.read_image(img_ID+'_001.npy', self.root)
        t1ce_img = self.read_image(img_ID+'_002.npy', self.root)
        flair_img = self.read_image(img_ID+'_003.npy', self.root)

        t1_img = t1_img[newaxis, :, :]
        t2_img = t2_img[newaxis, :, :]
        t1ce_img = t1ce_img[newaxis, :, :]
        flair_img = flair_img[newaxis, :, :]

        result = np.concatenate((t1_img,t2_img,t1ce_img,flair_img))
        return result

    def get_mask_file(self, fileName):
        result = fileName + '_004.npy'
        return result

    def __getitem__(self, index):
        # Select sample
        img_ID = self.fileList[index]
        mask_ID = self.get_mask_file(img_ID)

        # Load input and groundtruth
        img = self.read_train_image(img_ID, self.root)
        mask = self.read_image(mask_ID, self.root)
        return {
            'image':  torch.from_numpy(img).type(torch.FloatTensor),
            'mask':  torch.from_numpy(mask).type(torch.FloatTensor)
        }
