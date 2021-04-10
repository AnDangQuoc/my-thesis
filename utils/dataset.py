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
        img = img[newaxis,:,:]
        mask = self.read_image(mask_ID, self.root)
        return {
            'image':  torch.from_numpy(img).type(torch.FloatTensor),
            'mask':  torch.from_numpy(mask).type(torch.FloatTensor)
        }
