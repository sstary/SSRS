import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random, time
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from IPython.display import clear_output

from model.singleDino import UNetFormer as singleDino

from utils_dino import *

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

net = singleDino(num_classes=N_CLASSES).cuda()

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('All Params:   ', params)

# params1 = 0
# params2 = 0
# params3 = 0
# for name, param in net.image_encoder.named_parameters():
#     if "Adapter" not in name:
#         params1 += param.nelement()
#     else:
#         params2 += param.nelement()
# if MODEL == 'singleSAM':
#     for name, param in net.sam.prompt_encoder.named_parameters():
#         params3 += param.nelement()
#     for name, param in net.sam.mask_decoder.named_parameters():
#         params3 += param.nelement()
# print('ImgEncoder:   ', params1)
# print('Adapter:      ', params2)
# print('Others:       ', params-params1-params2-params3)

# for name, param in net.image_encoder.named_parameters():
#     if "Adapter" not in name:
#         params1 += param.nelement()
#     else:
#         params2 += param.nelement()
#
# print('ImgEncoder:   ', params1)
# print('Adapter:      ', params2)
# print('Others:       ', params-params1-params2-params3)

# def save_trainable_parameters_to_file(model, file_path="trainable_params.txt"):
#     with open(file_path, "w") as f:
#         f.write("Trainable parameters in the model:\n\n")
#         total_params = 0
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 f.write(f"{name:60} | shape: {tuple(param.shape)}\n")
#                 total_params += param.numel()
#         f.write(f"\nTotal trainable parameters: {total_params:,}\n")
# save_trainable_parameters_to_file(net, "trainable_params0.txt")

# def save_untrainable_parameters_to_file(model, file_path="untrainable_params.txt"):
#     with open(file_path, "w") as f:
#         f.write("Trainable parameters in the model:\n\n")
#         total_params = 0
#         for name, param in model.named_parameters():
#             if param.requires_grad == False:
#                 f.write(f"{name:60} | shape: {tuple(param.shape)}\n")
#                 total_params += param.numel()
#         f.write(f"\nTotal untrainable parameters: {total_params:,}\n")
# save_untrainable_parameters_to_file(net, "untrainable_params0.txt")

# Load the datasets
print("training : ", len(train_ids))
print("testing : ", len(test_ids))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)

    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    if DATASET == 'Hunan':
        eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64')) for id in test_ids)
    else:
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)

    all_preds = []
    all_gts = []
    # Switch the network to inference mode
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
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
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


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.5
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss = loss_calc(output, target, weights)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

        if e >= 0 and e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            net.eval()
            MIoU = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            if MIoU > MIoU_best:
                if DATASET == 'Vaihingen':
                    torch.save(net.state_dict(), './resultsv/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Potsdam':
                    torch.save(net.state_dict(), './resultsp/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Hunan':
                    torch.save(net.state_dict(), './resultsh/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                MIoU_best = MIoU
    print('MIoU_Best: ', MIoU_best)

if MODE == 'Train':
    train(net, optimizer, 50, scheduler)
elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('./resultsv/YOUR_MODEL'))
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsv/inference_{}_tile_{}_{}.png'.format(MODEL,id_,MIoU), img)

    elif DATASET == 'Potsdam':
        net.load_state_dict(torch.load('./resultsp/YOUR_MODEL'))
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=128)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsp/inference_{}_tile_{}_{}.png'.format(MODEL,id_,MIoU), img)

    elif DATASET == 'Hunan':
        net.load_state_dict(torch.load('./resultsh/YOUR_MODEL'))
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=256)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsh/inference_{}_tile_{}_{}.png'.format(MODEL,id_,MIoU), img)

