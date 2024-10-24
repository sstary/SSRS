import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
import random
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
import os
from IPython.display import clear_output
from models.swinfusenet.vision_transformer import SwinFuseNet as ViT_seg
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
    
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Parameters
# Parameters
WINDOW_SIZE = (224, 224) # Patch size
STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
# FOLDER = "D:/RA/CUHK-SZ/ISPRS_dataset/ISPRS_semantic_labeling_Vaihingen/"
FOLDER = "/media/lscsc/nas/xianping/ISPRS_dataset/Vaihingen/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10 # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memorycd x
# DATASET = 'Vaihingen'
# MAIN_FOLDER = FOLDER + 'Vaihingen/'
MAIN_FOLDER = FOLDER
DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.dsm_files = [DSM_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = io.imread(self.data_files[random_idx])

            data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[random_idx]
        else:
            # DSM is normalized in [0, 1]
            dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')

            min = np.min(dsm)
            max = np.max(dsm)
            dsm = (dsm - min) / (max - min)
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)
        # print((torch.from_numpy(dsm_p).shape))
        # print((torch.from_numpy(data_p).shape))
        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))
        
load_path= './pretrain/swin_tiny_patch4_window7_224.pth'
# load_path= '/content/drive/My Drive/SwinFuseNet/pretrain/swin_tiny_patch4_window7_224.pth'
net = ViT_seg(num_classes=6).to(device)
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('Params: ', params)
net.load_from(load_path)
# Load the datasets
#net = nn.DataParallel(net)
train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
test_ids = ['5', '21', '15', '30']

print('Tiles for training :', train_ids)
print('Tiles for testing :', test_ids)

train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 0.001
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)


def test(net, test_ids,num1,num2, all=False, stride=WINDOW_SIZE[0],batch_size=BATCH_SIZE,window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (convert_from_color(io.imread(LABEL_FOLDER.format(id))) for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    count = 0
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
            # e1 = 0
            # len1 = len(list(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))))

            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).to(device))

                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = Variable(torch.from_numpy(dsm_patches).to(device))

                # Do the inference
                outs = net(image_patches, dsm_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            clear_output()
            
            all_preds.append(pred)
            # all_gts.append(gt)
            all_gts.append(gt_e)

            clear_output()
            # Compute some metrics
            # metrics(pred.ravel(), gt_e.ravel())
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=2):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.to(device)

    # criterion = nn.NLLLoss2d(weight=weights)
    iter_ = 0
    acc_best = 88.0
    oa = 0
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            net.train()
            data, dsm, target = Variable(data.to(device)), Variable(dsm.to(device)), Variable(target.to(device))
            optimizer.zero_grad()
            output = net(data, dsm)
            loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                oa = accuracy(pred, gt)
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, oa))
                
            iter_ += 1

            del (data, target, loss)

        if e % 1 == 0:
            acc = test(net, test_ids, all=False, stride=32,num1=e,num2=batch_idx)
            oa = 0
            if acc > acc_best:
                torch.save(net.state_dict(), './res2/segnet256_epoch{}_{}_{}'.format(e,batch_idx, acc))
                acc_best = acc
    print("Train Done!!")

train(net, optimizer, 100, scheduler)

# ######   test   ####
# acc, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
# print("Acc: ", acc)
# for p, id_ in zip(all_preds, test_ids):
#     img = convert_to_color(p)
#     # plt.imshow(img) and plt.show()
#     io.imsave('./inference9064_tile_{}.png'.format(id_), img)
