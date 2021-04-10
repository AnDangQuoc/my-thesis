import os
import SimpleITK as sitk
import torch
import numpy as np

RAW_DATA_PATH = './train_data'
DUMP_DATA_PATH = './data/train'

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


def gen_file_name(file_type, layer_index):
    return f'BRATS_{FILE_COUNT:03d}_{layer_index:03d}_{IMAGE_TYPE_MAP[file_type]}'


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

    seg_img_torch = torch.from_numpy(seg_img).type(torch.float32)

    for i in range(LAYER_SIZE):
        label_list = list(seg_img_torch[i].unique())
        if(len(label_list) <= 1):
            continue
        t1_img_name = gen_file_name('t1', i)
        t2_img_name = gen_file_name('t2', i)
        t1ce_img_name = gen_file_name('t1ce', i)
        flair_img_name = gen_file_name('flair', i)
        seg_img_name = gen_file_name('seg', i)

        np.save(os.path.join(DUMP_DATA_PATH, t1_img_name), t1_img[i])
        np.save(os.path.join(DUMP_DATA_PATH, t2_img_name), t2_img[i])
        np.save(os.path.join(DUMP_DATA_PATH, t1ce_img_name), t1ce_img[i])
        np.save(os.path.join(DUMP_DATA_PATH, flair_img_name), flair_img[i])
        np.save(os.path.join(DUMP_DATA_PATH, seg_img_name), seg_img[i])


def main():
    global FILE_COUNT
    global FILE_NAME
    HGG_PATH = os.path.join(RAW_DATA_PATH, 'HGG')
    LGG_PATH = os.path.join(RAW_DATA_PATH, 'LGG')

    HGG_LIST = get_file(HGG_PATH)
    LGG_LIST = get_file(LGG_PATH)
    for i in HGG_LIST:
        if i.startswith('.'):
            continue
        handle_image(HGG_PATH, i)
        FILE_NAME.append(f'BRATS_{FILE_COUNT:05d}')
        FILE_COUNT = FILE_COUNT + 1

    for i in LGG_LIST:
        if i.startswith('.'):
            continue
        handle_image(LGG_PATH, i)
        FILE_NAME.append(f'BRATS_{FILE_COUNT:05d}')
        FILE_COUNT = FILE_COUNT + 1

    file1 = open("./dev.txt", "a")
    for i in FILE_NAME:
        file1.write(i)
    file1.close()


main()
