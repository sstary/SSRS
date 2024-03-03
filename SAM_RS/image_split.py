import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile
from PIL import Image

def normalize(img):
    min = img.min()
    max = img.max()
    x = 255.0 * (img - min) / (max - min)
    return x


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


split_width = 480
split_height = 480


#####ISPRSDATA#####################################################


# overlap=0.2
overlap=0

# path = r"ori_V/ori_train/dsm/"
# list = []
# filelist = os.listdir(path)
# for file in filelist:
#     jpg_name = os.path.join(path, file)  # 拼接图像的读取地址
#     if '.tif' in jpg_name:
#         count = 0
#         print(jpg_name)
#         img = cv2.imread(jpg_name)
#         img=np.array(img)
#         img_h, img_w,_ = img.shape
#         X_points = start_points(img_w, split_width, overlap)
#         Y_points = start_points(img_h, split_height, overlap)
#
#         name = 'remote_all_V/dsm/train'
#         frmt = '.tif'
#         ind = str(int(file.split('_')[2]))  + '_' + str(int(file.split('_')[3]))
#         list.append(ind)
#
#         splits = []
#         for i in Y_points:
#             for j in X_points:
#                 split = img[i:i + split_height, j:j + split_width]
#                 cv2.imwrite(os.path.join(name, ind + '_' + str(count) + frmt), split)
#                 count += 1
#                 splits.append(split)
#         print(count)

## Potsdam
# ids = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7', '4_12', '6_8', '6_12', '6_7', '4_11']
# ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']

# Vaihingen
ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
ids = ['5', '21', '15', '30']

path = r"ISPRS_dataset/Potsdam/1_DSM_normalisation/"
list = []
filelist = os.listdir(path)
for file in filelist:
    if '.jpg' in file:
        jpg_name = os.path.join(path, file)  # 拼接图像的读取地址
        # ind = str(int(file.split('_')[3].replace('area', '').replace('.tif', ''))) # extract vaihingen
        ind = str(file.replace('dsm_potsdam_', '').replace('_normalized_lastools.jpg', '')) # extract potsdam
        if ind in ids:
            count = 0
            print(jpg_name)
            img = cv2.imread(jpg_name)
            img = np.array(img)
            img_h, img_w, _ = img.shape
            # img_h, img_w = img.shape
            X_points = start_points(img_w, split_width, overlap)
            Y_points = start_points(img_h, split_height, overlap)

            name = '../MANet/Potsdam/train/dsm/'
            frmt = '.tif'
            # ind = str(int(file.split('_')[2])) + '_' + str(int(file.split('_')[3]))
            # ind = str(int(file.split('_')[3].replace('area', '').replace('.tif', '')))
            list.append(ind)

            splits = []
            for i in Y_points:
                for j in X_points:
                    split = img[i:i + split_height, j:j + split_width]
                    split = split[:, :, 0]
                    # cropped = np.zeros([256, 256, 3]).astype(np.uint8)
                    # for p in range(0, 256):
                    #     for q in range(0, 256):
                    #         cropped[p, q, 0] = split[p, q]
                    #         cropped[p, q, 1] = split[p, q]
                    #         cropped[p, q, 2] = split[p, q]
                    # tifffile.imwrite(os.path.join(name, ind + '_' + str(count) + frmt), split)
                    cv2.imwrite(os.path.join(name, ind + '_' + str(count) + '.tif'), split)
                    count += 1
                    splits.append(split)
            print(count)
