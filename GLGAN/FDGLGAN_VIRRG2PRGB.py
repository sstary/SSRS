import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
import torch.utils.data as data
import torch.optim as optim
import torch.nn.init
from torch.autograd import Variable
from IPython.display import clear_output
import os
import time
from utils import *
from transdiscri import transDiscri
from FTUNetFormer_11 import ft_unetformer as ViT_seg
from func import loss_calc, bce_loss
from loss import entropy_loss
from func import prob_2_entropy
import torch.backends.cudnn as cudnn


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener


# Parameters
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
WINDOW_SIZE = (256, 256) # Patch size


STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "/media/lscsc/nas/xianping/ISPRS_dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10 # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

MAIN_FOLDER_P = FOLDER + 'Vaihingen/'
DATA_FOLDER_P = MAIN_FOLDER_P + 'top/top_mosaic_09cm_area{}.tif'
LABEL_FOLDER_P = MAIN_FOLDER_P + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
ERODED_FOLDER_P = MAIN_FOLDER_P + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

MAIN_FOLDER_V = FOLDER + 'Potsdam/'
DATA_FOLDER_V = MAIN_FOLDER_V + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
LABEL_FOLDER_V = MAIN_FOLDER_V + '5_Labels_for_participants/top_potsdam_{}_label.tif'
ERODED_FOLDER_V = MAIN_FOLDER_V + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'

# net = ResUnetPlusPlus(3).cuda()

model = ViT_seg(num_classes=N_CLASSES)
params = 0
for name, param in model.named_parameters():
    params += param.nelement()
print(params)
# saved_state_dict = torch.load('2_Advent/pretrained/DeepLab_resnet_pretrained_imagenet.pth')
# new_params = model.state_dict().copy()
# for i in saved_state_dict:
#     i_parts = i.split('.')
#     if not i_parts[1] == 'layer5':
#         new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
# model.load_state_dict(new_params)
model.train()
model = model.cuda()
cudnn.benchmark = True
cudnn.enabled = True
d_aux = transDiscri(num_classes=N_CLASSES)
d_aux.train()
d_aux.cuda()

# seg maps, i.e. output, level
d_main = transDiscri(num_classes=N_CLASSES)
d_main.train()
d_main.cuda()

train_ids_P = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
test_ids_P = ['5', '21', '15', '30']
train_ids_V = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
             '4_12', '6_8', '6_12', '6_7', '4_11']
test_ids_V = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']

print("Vaihingen for training : ", train_ids_P)
print("Vaihingen for testing : ", test_ids_P)
print("Potsdam for training : ", train_ids_V)
print("Potsdam for testing : ", test_ids_V)
DATASET_P = 'Vaihingen'
DATASET_V = 'Potsdam'
train_set = ISPRS_dataset(train_ids_P, train_ids_V, DATASET_P, DATASET_V, DATA_FOLDER_P, DATA_FOLDER_V,
                          LABEL_FOLDER_P, LABEL_FOLDER_V, cache=CACHE, RGB_flag=True)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LEARNING_RATE_D = 1e-4
LAMBDA_ADV_MAIN = 0.001
LAMBDA_ADV_AUX = 0.0002
LAMBDA_DECOMP = 0.01
LAMBDA_SEG_MAIN = 1.0
LAMBDA_SEG_AUX = 0.1
print('LAMBDA_DECOMP: ', LAMBDA_DECOMP)
# optimizer = optim.SGD(model.optim_parameters(LEARNING_RATE),
#                         lr=LEARNING_RATE,
#                         momentum=MOMENTUM,
#                         weight_decay=WEIGHT_DECAY)
optimizer = optim.SGD(model.parameters(),
                        lr=LEARNING_RATE,
                        momentum=MOMENTUM,
                        weight_decay=WEIGHT_DECAY)

# discriminators' optimizers
optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=LEARNING_RATE_D,
                                betas=(0.9, 0.99))
optimizer_d_main = optim.Adam(d_main.parameters(), lr=LEARNING_RATE_D,
                                betas=(0.9, 0.99))

# labels for adversarial training
source_label = 0
target_label = 1

def test(test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER_V.format(id))[:, :, :3], dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER_V.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(LABEL_FOLDER_V.format(id))) for id in test_ids)

    # eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    model.eval()
    with torch.no_grad():
        for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):

                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                # Do the inference
                _, pred2, _, _= model(image_patches)
                outs = F.softmax(pred2, dim=1)
                outs = outs.data.cpu().numpy()

                # Fill in the results array-
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            clear_output()
            all_preds.append(pred)
            all_gts.append(gt_e)

            clear_output()
            # Compute some metrics
            # metrics(pred.ravel(), gt_e.ravel())
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    model.train()
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(epochs, weights=WEIGHTS, save_epoch=2):
    weights = weights.cuda()
    MIoU_best = 0.55
    iter = 0
    model.train()
    d_aux.train()
    d_main.train()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for batch_idx, (images, labels, images_t, labels_t) in enumerate(train_loader):
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))

            optimizer_d_aux.zero_grad()
            optimizer_d_main.zero_grad()
            adjust_learning_rate_D(optimizer_d_aux, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))
            adjust_learning_rate_D(optimizer_d_main, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))

            # train G
            # don't accumulate grads in D
            for param in d_aux.parameters():
                param.requires_grad = False
            for param in d_main.parameters():
                param.requires_grad = False

            # train with source
            # x2, x1, f_di, f_ds
            images = Variable(images).cuda()
            pred1, pred2, f_dix, f_dsx = model(images)
            loss_seg1 = loss_calc(pred1, labels, weights)
            loss_seg2 = loss_calc(pred2, labels, weights)
            
            images_t = Variable(images_t).cuda()
            pred_target1, pred_target2, f_diy, f_dsy = model(images_t)
            D_out1 = d_aux(prob_2_entropy(F.softmax(pred_target1, dim=1)))
            D_out2 = d_main(prob_2_entropy(F.softmax(pred_target2, dim=1)))
            loss_adv_target1 = bce_loss(D_out1, source_label)
            loss_adv_target2 = bce_loss(D_out2, source_label)
            loss_base = multi_discrepancy(f_dix, f_diy)
            loss_detail = multi_discrepancy(f_dsx, f_dsy)
            loss = (LAMBDA_SEG_MAIN * loss_seg2
                + LAMBDA_SEG_AUX * loss_seg1
                + LAMBDA_ADV_MAIN * loss_adv_target2
                + LAMBDA_ADV_AUX * loss_adv_target1
                + LAMBDA_DECOMP * (loss_base + loss_detail))
            loss.backward()

            # train D
            # bring back requires_grad
            for param in d_aux.parameters():
                param.requires_grad = True

            for param in d_main.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred1.detach()
            pred2 = pred2.detach()
            D_out1 = d_aux(prob_2_entropy(F.softmax(pred1, dim=1)))
            D_out2 = d_main(prob_2_entropy(F.softmax(pred2, dim=1)))
            loss_D1 = bce_loss(D_out1, source_label)
            loss_D2 = bce_loss(D_out2, source_label)
            loss_D1 = loss_D1 / 2
            loss_D2 = loss_D2 / 2
            loss_D1.backward()
            loss_D2.backward()

            # train with target
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()
            D_out1 = d_aux(prob_2_entropy(F.softmax(pred_target1, dim=1)))
            D_out2 = d_main(prob_2_entropy(F.softmax(pred_target2, dim=1)))
            loss_D1 = bce_loss(D_out1, target_label)
            loss_D2 = bce_loss(D_out2, target_label)
            loss_D1 = loss_D1 / 2
            loss_D2 = loss_D2 / 2
            loss_D1.backward()
            loss_D2.backward()

            optimizer.step()
            optimizer_d_aux.step()
            optimizer_d_main.step()
            if iter % 100 == 0:
                clear_output()
                pred = np.argmax(pred_target2.data.cpu().numpy()[0], axis=0)
                gt = labels_t.data.cpu().numpy()[0]
                end_time = time.time()
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)] lr: {:.12f} lr_D: {:.12f} Loss: {:.6f} Loss_Base: {:.6f} Loss_Detail: {:.6f} Loss_D1: {:.6f} Loss_D2: {:.6f} Accuracy: {:.2f}% Timeuse: {:.2f}'.format(
                    epoch, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'], optimizer_d_aux.state_dict()['param_groups'][0]['lr'],
                    loss_seg2.data, loss_base.data, loss_detail.data, loss_D1.data, loss_D2.data, accuracy(pred, gt), end_time - start_time))
                start_time = time.time()
            iter += 1
            del (images, labels, images_t, labels_t, loss, loss_base, loss_detail, loss_D1, loss_D2)

        if epoch % 1 == 0:
            # We validate with the largest possible stride for faster computing
            start_time = time.time()
            MIoU = test(test_ids_V, all=False, stride=128)
            end_time = time.time()
            print('Test Stide_128 time use: ', end_time - start_time)
            start_time = time.time()
            if MIoU > MIoU_best:
                torch.save(model.state_dict(), './Train_Model/UNetFormer_V2P_epoch{}_{}'.format(epoch, MIoU))
                MIoU_best = MIoU

    print("Train Done!!")

# train(100)

######   test   ####
model.load_state_dict(torch.load('./Train_Model/UNetFormer_V2P_epoch63_0.6141165054038286'))
acc, all_preds, all_gts = test(test_ids_V, all=True, stride=32)
print("Acc: ", acc)
for p, id_ in zip(all_preds, test_ids_V):
   img = convert_to_color(p)
   io.imsave('./Test_Vision/V2P_' + str(acc) + '_tile_{}.png'.format(id_), img)
