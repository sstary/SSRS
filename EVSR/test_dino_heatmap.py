import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random, time
import sys
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from IPython.display import clear_output

from model.singleDino_heatmap import UNetFormer as singleDino

from utils_dino import *

MODE = 'Test'

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

net = singleDino(num_classes=N_CLASSES).cuda()

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('All Params:   ', params)

print("training : ", len(train_ids))
print("testing : ", len(test_ids))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)


def save_heatmap(image_patch, label_patch, heatmaps, index, point=(100, 65)):
    x_comp = point[1]
    y_comp = point[0]
    image_patch = np.asarray(255 * image_patch, dtype='uint8').transpose((1, 2, 0))
    heatmaps = [cv2.resize(h, (256, 256)) for h in heatmaps]
    heatmaps = [np.uint8(255 * h) for h in heatmaps]
    heatmaps = [cv2.applyColorMap(h, cv2.COLORMAP_JET)[:, :, (2, 1, 0)] for h in heatmaps]

    fig = plt.figure(figsize=(32, 4))
    fig.add_subplot(1, 10, 1)
    plt.imshow(image_patch)
    plt.gca().add_patch(plt.Rectangle((x_comp - 2, y_comp - 2), 2, 2, color='red', fill=False, linewidth=1))
    plt.axis('off')

    for i, heatmap in enumerate(heatmaps):
        fig.add_subplot(1, 10, i + 2)
        plt.imshow(heatmap)
        plt.gca().add_patch(plt.Rectangle((x_comp - 2, y_comp - 2), 2, 2, color='red', fill=False, linewidth=1))
        plt.axis('off')

    fig.add_subplot(1, 10, 10)
    plt.imshow(label_patch)
    plt.gca().add_patch(plt.Rectangle((x_comp - 2, y_comp - 2), 2, 2, color='red', fill=False, linewidth=1))
    plt.axis('off')
    clear_output()
    plt.savefig('heatmap_singleDino2_' + str(index) + '.pdf', dpi=1200)
    plt.close(fig)


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)

    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    if DATASET == 'Hunan':
        eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64')) for id in test_ids)
    else:
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)

    all_preds = []
    all_gts = []
    for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        index = 0
        for i, coords in enumerate(
                tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                    leave=False)):
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), requires_grad=True)

            outs, heatmaps = net(image_patches, mode='Test')
            outs = outs.data.cpu().numpy()

            gt_patch = np.copy(gt[coords[0][0]:coords[0][0] + coords[0][2], coords[0][1]:coords[0][1] + coords[0][3]])
            save_heatmap(image_patches[0].detach().cpu().numpy(), gt_patch, heatmaps, index)
            index += 1

            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out
            del (outs)

        pred = np.argmax(pred, axis=-1)
        all_preds.append(pred)
        all_gts.append(gt_e)
        clear_output()

    if DATASET == 'Hunan':
        accuracy = metrics_hunan(np.concatenate([p.ravel() for p in all_preds]),
                        np.concatenate([p.ravel() for p in all_gts]).ravel())
    else:
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                        np.concatenate([p.ravel() for p in all_gts]).ravel())

    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy

if MODE == 'Test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('./resultsv/YOUR_MODEL'))
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)

    elif DATASET == 'Potsdam':
        net.load_state_dict(torch.load('./resultsp/YOUR_MODEL'))
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=128)

    elif DATASET == 'Hunan':
        net.load_state_dict(torch.load('./resultsh/YOUR_MODEL'))
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=256)
