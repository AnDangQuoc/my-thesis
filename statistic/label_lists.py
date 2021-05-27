import os
import SimpleITK as sitk
import torch
import numpy as np
import json

RAW_DATA_PATH = './train_data'
DUMP_DATA_PATH = './data/train'

LAYER_SIZE = 155
IMAGE_SIZE = 240

TOTAL_PIXELS = IMAGE_SIZE*IMAGE_SIZE

FILE_COUNT = 0

IMAGE_TYPE_MAP = {
    't1': '000',
    't2': '001',
    't1ce': '002',
    'flair': '003',
    'seg': '004'
}

FILE_NAME = []


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_file(file_path):
    file_list = os.listdir(file_path)
    return file_list


# Image dimension z,y,x
def read_image(image_path):
    img = sitk.ReadImage(image_path)
    img = sitk.GetArrayFromImage(img)
    return img


def gen_file_name(file_type, layer_index):
    return f'BRATS_{FILE_COUNT:03d}_{layer_index:03d}_{IMAGE_TYPE_MAP[file_type]}'


def handle_image(fileDir, fileName, stat={}):
    print('Processing file '+fileName)

    seg_img_path = os.path.join(fileDir, fileName, fileName+'_seg.nii.gz')

    # Read file to numpy
    seg_img = read_image(seg_img_path)

    label_lists = stat["label_lists"]

    coverage = {}

    if not label_lists:
        label_lists = []

    for i in range(LAYER_SIZE):
        labels = np.unique(seg_img[i])
        if(len(labels) <= 1):
            continue
        for x in labels:
            if str(x) not in label_lists:
                label_lists.append(str(x))

            if str(x) != "0":
                if i not in coverage:
                    coverage[i] = {"1": {}, "2": {}, "3": {}, "4": {}}
                num_pixel = np.count_nonzero(seg_img[i] == x)
                coverage[i][str(x)]['num_pixel'] = num_pixel
                coverage[i][str(x)]['coverage'] = str(
                    num_pixel / TOTAL_PIXELS * 100)

    stat["label_lists"] = label_lists
    stat["coverage"][fileName] = coverage


def main():
    HGG_PATH = os.path.join(RAW_DATA_PATH, 'HGG')
    LGG_PATH = os.path.join(RAW_DATA_PATH, 'LGG')

    HGG_LIST = get_file(HGG_PATH)
    LGG_LIST = get_file(LGG_PATH)

    stat = {
        "label_lists": [],
        "coverage": {}
    }

    COUNT_IMG = 0
    for i in HGG_LIST:
        # if COUNT_IMG > 10:
        #     break

        if i.startswith('.'):
            continue
        handle_image(HGG_PATH, i, stat)

        # COUNT_IMG = COUNT_IMG + 1

    COUNT_IMG = 0
    for i in LGG_LIST:

        # if COUNT_IMG > 10:
        #     break
        
        if i.startswith('.'):
            continue
        handle_image(LGG_PATH, i, stat)

        # COUNT_IMG = COUNT_IMG + 1

    file1 = open("./statistic/result.json", "w")
    file1.write(json.dumps(stat))
    file1.close()


main()
