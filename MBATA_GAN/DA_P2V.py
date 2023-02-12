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
from utils import *
from discriminator_enhance import FCDiscriminator
from discriminator_feature import FCDiscriminator_feature
from deeplab_enhanceCross import DeeplabMulti
import time
import torch.autograd as autograd
import cv2
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
FOLDER = "../ISPRS_dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10 # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

MAIN_FOLDER_P = FOLDER + 'Potsdam/'
DATA_FOLDER_P = MAIN_FOLDER_P + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
LABEL_FOLDER_P = MAIN_FOLDER_P + '5_Labels_for_participants/top_potsdam_{}_label.tif'
ERODED_FOLDER_P = MAIN_FOLDER_P + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'

MAIN_FOLDER_V = FOLDER + 'Vaihingen/'
DATA_FOLDER_V = MAIN_FOLDER_V + 'top/top_mosaic_09cm_area{}.tif'
LABEL_FOLDER_V = MAIN_FOLDER_V + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
ERODED_FOLDER_V = MAIN_FOLDER_V + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

# net = ResUnetPlusPlus(3).cuda()
net = DeeplabMulti(num_classes=N_CLASSES)
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('Params: ', params)
model_D1 = FCDiscriminator(num_classes=N_CLASSES)
model_D2 = FCDiscriminator(num_classes=N_CLASSES)
model_Df = FCDiscriminator_feature()

net = net.cuda()
model_D1 = model_D1.cuda()
model_D2 = model_D2.cuda()
model_Df = model_Df.cuda()

train_ids_P = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
             '4_12', '6_8', '6_12', '6_7', '4_11']
test_ids_P = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
train_ids_V = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
test_ids_V = ['5', '21', '15', '30']

print("Potsdam for training : ", train_ids_P)
print("Potsdam for testing : ", test_ids_P)
print("Vaihingen for training : ", train_ids_V)
print("Vaihingen for testing : ", test_ids_V)
DATASET_P = 'Potsdam'
DATASET_V = 'Vaihingen'
train_set = ISPRS_dataset(train_ids_P, train_ids_V, DATASET_P, DATASET_V, DATA_FOLDER_P, DATA_FOLDER_V,
                          LABEL_FOLDER_P, LABEL_FOLDER_V, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
LAMBDA_ADV_DF = 0.005
LAMBDA_MMD = 2.0
if RESTORE_FROM[:4] == 'http':
    saved_state_dict = model_zoo.load_url(RESTORE_FROM)
else:
    saved_state_dict = torch.load(RESTORE_FROM)

new_params = net.state_dict().copy()
for i in saved_state_dict:
    # Scale.layer5.conv2d_list.3.weight
    i_parts = i.split('.')
    # print i_parts
    if not N_CLASSES == 6 or not i_parts[1] == 'layer5':
        new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        # print i_parts
net.load_state_dict(new_params)

optimizer = optim.SGD(net.optim_parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
optimizer.zero_grad()

optimizer_D1 = optim.Adam(model_D1.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
optimizer_D2 = optim.Adam(model_D2.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
optimizer_Df = optim.Adam(model_Df.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
optimizer_D1.zero_grad()
optimizer_D2.zero_grad()
optimizer_Df.zero_grad()

bce_loss = torch.nn.BCEWithLogitsLoss()
interp = nn.Upsample(size=(256, 256), mode='bilinear')

source_label = 0
target_label = 1

def test(test_ids_P, test_ids_V, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_imagess = (1 / 255 * np.asarray(io.imread(DATA_FOLDER_P.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids_P)
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER_V.format(id)), dtype='float32') for id in test_ids_V)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER_V.format(id)), dtype='uint8') for id in test_ids_V)
    eroded_labels = (convert_from_color(io.imread(LABEL_FOLDER_V.format(id))) for id in test_ids_V)

    # eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    net.eval()
    feature_map = False
    with torch.no_grad():
        for imgs, imgt, gt, gt_e in tqdm(zip(test_imagess, test_images, test_labels, eroded_labels), total=len(test_ids_V), leave=False):
            pred = np.zeros(imgt.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(imgt, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(imgt, step=stride, window_size=window_size)), total=total,
                            leave=False)):

                # Build the tensor
                coordss = ((100, 100, 256, 256),)
                # coordss = ((100, 100, 256, 256))
                image_patchess = [np.copy(imgs[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coordss]
                image_patchess = np.asarray(image_patchess)
                image_patchess = Variable(torch.from_numpy(image_patchess).cuda())
                image_patches = [np.copy(imgt[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda())

                # Do the inference
                _, _, _, pred_target2, _, atty, _, _ = net(image_patchess, image_patches)
                pred2 = F.softmax(pred_target2, dim=1)
                outs = interp(pred2)
                if feature_map:
                    # x_comp = 80
                    # y_comp = 20
                    # pred = outs[:, 1, x_comp, y_comp]

                    x_comp = 50
                    y_comp = 100
                    pred = outs[:, 4, x_comp, y_comp]
                    feature = outs
                    feature_grad = autograd.grad(pred, feature, allow_unused=True, retain_graph=True)[0]
                    
                    grads = feature_grad  # 获取梯度
                    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
                    # 此处batch size默认为1，所以去掉了第0维（batch size维）
                    pooled_grads = pooled_grads[0]
                    feature = feature[0]
                    # print("pooled_grads:", pooled_grads.shape)
                    # print("feature:", feature.shape)
                    # feature.shape[0]是指定层feature的通道数
                    for i in range(feature.shape[0]):
                        feature[i, ...] *= pooled_grads[i, ...]

                    heatmap = feature.detach().cpu().numpy()
                    heatmap = np.mean(heatmap, axis=0)
                    heatmap1 = np.maximum(heatmap, 0)
                    heatmap1 /= np.max(heatmap1)
                    heatmap1 = cv2.resize(heatmap1, (256, 256))
                    # heatmap[heatmap < 0.7] = 0
                    heatmap1 = np.uint8(255 * heatmap1)
                    heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
                    heatmap1 = heatmap1[:, :, (2, 1, 0)]

                    fig = plt.figure()
                    fig.add_subplot(1, 2, 1)
                    image_patches = np.asarray(255 * torch.squeeze(image_patches).cpu(), dtype='uint8').transpose((1, 2, 0))
                    plt.imshow(image_patches)
                    plt.axis('off')
                    plt.gca().add_patch(plt.Rectangle((x_comp - 2, y_comp - 2), 2, 2, color='red', fill=False, linewidth=1))

                    fig.add_subplot(1, 2, 2)
                    plt.imshow(heatmap1)
                    plt.gca().add_patch(plt.Rectangle((x_comp - 2, y_comp - 2), 2, 2, color='red', fill=False, linewidth=1))
                    plt.axis('off')
                    plt.savefig('CSATAGAN_car.pdf', dpi=1200)
                    plt.show()

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
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    net.train()
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy

def train(epochs, weights=WEIGHTS, save_epoch=2):
    weights = weights.cuda()
    MIoU_best = 0.55
    # MIoU_best = 0
    Name_best = ''
    iter = 0
    net.train()
    model_D1.train()
    model_D2.train()
    model_Df.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (images, labels, images_t, labels_t) in enumerate(train_loader):
            start_time = time.time()
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))

            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            optimizer_Df.zero_grad()
            adjust_learning_rate_D(optimizer_D1, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))
            adjust_learning_rate_D(optimizer_D2, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))
            adjust_learning_rate_D(optimizer_Df, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))
            # train G
            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False
            for param in model_Df.parameters():
                param.requires_grad = False

            # train with source and target
            images = Variable(images).cuda()
            images_t = Variable(images_t).cuda()
            pred1, pred2, pred_target1, pred_target2, attx, atty, attxp, attyp = net(images, images_t)

            loss_mmd = discrepancy(attxp, attyp)
            ## coral
            # loss_mmd = coral(attxp, attyp)
            pred1 = interp(pred1)
            pred2 = interp(pred2)
            loss_seg1 = loss_calc(pred1, labels, weights)
            loss_seg2 = loss_calc(pred2, labels, weights)
            pred_target1 = interp(pred_target1)
            pred_target2 = interp(pred_target2)
            D_out1 = model_D1(F.softmax(pred_target1, dim=1))
            D_out2 = model_D2(F.softmax(pred_target2, dim=1))
            D_f = model_Df(F.softmax(atty, dim=1))
            loss_adv_target1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
            loss_adv_target2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())
            loss_adv_targetf = bce_loss(D_f, Variable(torch.FloatTensor(D_f.data.size()).fill_(source_label)).cuda())
            loss = loss_seg2 + LAMBDA_SEG * loss_seg1 + LAMBDA_ADV_TARGET1 * loss_adv_target1 + \
                   LAMBDA_ADV_TARGET2 * loss_adv_target2 + LAMBDA_ADV_DF * loss_adv_targetf + LAMBDA_MMD * loss_mmd
            loss.backward()
            # train D
            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True
            for param in model_Df.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred1.detach()
            pred2 = pred2.detach()
            attx = attx.detach()
            D_out1 = model_D1(F.softmax(pred1, dim=1))
            D_out2 = model_D2(F.softmax(pred2, dim=1))
            D_f = model_Df(F.softmax(attx, dim=1))
            loss_D1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
            loss_D2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())
            loss_Df = bce_loss(D_f, Variable(torch.FloatTensor(D_f.data.size()).fill_(source_label)).cuda())
            loss_D1 = loss_D1 / 2
            loss_D2 = loss_D2 / 2
            loss_D1.backward()
            loss_D2.backward()
            loss_Df.backward()

            # train with target
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()
            atty = atty.detach()
            D_out1 = model_D1(F.softmax(pred_target1, dim=1))
            D_out2 = model_D2(F.softmax(pred_target2, dim=1))
            D_f = model_Df(F.softmax(atty, dim=1))
            loss_D1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())
            loss_D2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda())
            loss_Df = bce_loss(D_f, Variable(torch.FloatTensor(D_f.data.size()).fill_(target_label)).cuda())
            loss_D1 = loss_D1 / 2
            loss_D2 = loss_D2 / 2
            loss_D1.backward()
            loss_D2.backward()
            loss_Df.backward()

            optimizer.step()
            optimizer_D1.step()
            optimizer_D2.step()
            optimizer_Df.step()
            if iter % 1 == 0:
                clear_output()
                pred = np.argmax(pred_target2.data.cpu().numpy()[0], axis=0)
                gt = labels_t.data.cpu().numpy()[0]
                end_time = time.time()
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)] lr: {:.12} lr_D: {:.12} Loss: {:.6} Loss_seg: {:.6} Loss_adv: {:.6} Loss_mmd: {:.6} Loss_D1: {:.6} Loss_D2: {:.6} Accuracy: {:.2f}% Timeuse: {:.2f}'.format(
                    epoch, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'], optimizer_D1.state_dict()['param_groups'][0]['lr'],
                    loss.data, loss_seg2.data, loss_adv_target2.data, loss_mmd.data, loss_D1.data, loss_D2.data, accuracy(pred, gt), end_time - start_time))
                start_time = time.time()
            iter += 1
            del (images, labels, images_t, labels_t, loss, loss_D1, loss_D2)

            if iter % 100 == 0:
                # We validate with the largest possible stride for faster computing
                start_time = time.time()
                MIoU = test(test_ids_P, test_ids_V, all=False, stride=32)
                end_time = time.time()
                print('Test Stide_32 time use: ', end_time - start_time)
                start_time = time.time()
                if MIoU > MIoU_best:
                    torch.save(net.state_dict(), './Train_Model/DATrans_P2V_epoch{}_{}'.format(epoch, MIoU))
                    MIoU_best = MIoU
    print("Train Done!!")

train(50)

###  Test  ####
# net.load_state_dict(torch.load('Train_Model/'))
# start = time.time()
# acc, all_preds, all_gts = test(test_ids_P, test_ids_V, all=True, stride=32)
# print('Test tride time use: ', time.time() - start)
# print("Acc: ", acc)
# for p, id_ in zip(all_preds, test_ids_V):
#     img = convert_to_color(p)
#     io.imsave('./Test_Vision/P2V_tile_{}.png'.format(id_), img)