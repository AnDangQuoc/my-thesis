import os
import SimpleITK as sitk
import torch
import numpy as np
import json

RAW_DATA_PATH = ''
DUMP_DATA_PATH = ''

LAYER_SIZE = 155
IMAGE_SIZE = 240

FILE_COUNT = 0

IMAGE_TYPE_MAP = {
    't1': '000',
    't2': '001',
    't1ce': '002',
    'flair': '003',
    'seg': '004'
}

FILE_NAME = []


def get_file(file_path):
    file_list = os.listdir(file_path)
    return file_list


# Image dimension z,y,x
def read_image(image_path):
    img = sitk.ReadImage(image_path)
    img = sitk.GetArrayFromImage(img)
    return img


def gen_file_name(file_type, layer_index, fileName):
    return f'{fileName}_{layer_index:03d}_{IMAGE_TYPE_MAP[file_type]}'


def handle_image(fileDir, fileName):
    print('Processing file '+fileName)

    t1_img_path = os.path.join(fileDir, fileName, fileName + '_t1.nii.gz')
    t1ce_img_path = os.path.join(fileDir, fileName, fileName + '_t1ce.nii.gz')
    t2_img_path = os.path.join(fileDir, fileName, fileName + '_t2.nii.gz')
    flair_img_path = os.path.join(
        fileDir, fileName, fileName + '_flair.nii.gz')

    seg_img_path = os.path.join(fileDir, fileName, fileName+'_seg.nii.gz')

    # Read file to numpy
    t1_img = read_image(t1_img_path)
    t1ce_img = read_image(t1ce_img_path)
    t2_img = read_image(t2_img_path)
    flair_img = read_image(flair_img_path)
    seg_img = read_image(seg_img_path)

    seg_img_torch = torch.from_numpy(seg_img.astype(int)).type(torch.float32)

    for i in range(LAYER_SIZE):
        label_list = list(seg_img_torch[i].unique())
        if(len(label_list) <= 1):
            continue
        t1_img_name = gen_file_name('t1', i, fileName)
        t2_img_name = gen_file_name('t2', i, fileName)
        t1ce_img_name = gen_file_name('t1ce', i, fileName)
        flair_img_name = gen_file_name('flair', i, fileName)
        seg_img_name = gen_file_name('seg', i, fileName)

        np.save(os.path.join(DUMP_DATA_PATH,
                f'data/{t1_img_name}'), t1_img[i])
        np.save(os.path.join(DUMP_DATA_PATH,
                f'data/{t2_img_name}'), t2_img[i])
        np.save(os.path.join(DUMP_DATA_PATH,
                f'data/{t1ce_img_name}'), t1ce_img[i])
        np.save(os.path.join(DUMP_DATA_PATH,
                f'data/{flair_img_name}'), flair_img[i])
        np.save(os.path.join(DUMP_DATA_PATH,
                f'data/{seg_img_name}'), seg_img[i])

        FILE_NAME.append(f'{fileName}_{i:03d}')

def main():
    global FILE_COUNT
    global FILE_NAME

    FILE_LIST = get_file(RAW_DATA_PATH)
    for i in FILE_LIST:
        if i.startswith('.'):
            continue
        handle_image(RAW_DATA_PATH, i)

    file1 = open(os.path.join(DUMP_DATA_PATH, 'stackTrain.json'), "w")
    file1.write(json.dumps(FILE_NAME))
    file1.close()


main()
