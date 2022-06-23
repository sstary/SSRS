# -*- coding: utf-8 -*-
# @Author  : Haonan Wang
# @File    : CTrans.py
# @Software: PyCharm
# coding=utf-8
import numpy as np
from skimage import io
from tqdm import tqdm_notebook as tqdm
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
import os
from Utils.utils import *
from Utils.CMFNet import *
from IPython.display import clear_output
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

# Parameters
WINDOW_SIZE = (256, 256) # Patch size
STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "./ISPRS_dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10 # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory
config_vit = config.get_CTranS_config()
DATASET = 'Vaihingen'

if DATASET == 'Potsdam':
    MAIN_FOLDER = FOLDER + 'Potsdam/'
    # Uncomment the next line for IRRG data
    # DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG/top_potsdam_{}_IRRG.tif'
    # For RGB data
    DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
elif DATASET == 'Vaihingen':
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
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

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))

# instantiate the network
net = CMFNet()

# Download VGG-16 weights from PyTorch
vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
if not os.path.isfile('../vgg16_bn-6c64b313.pth'):
    weights = URLopener().retrieve(vgg_url, '../vgg16_bn-6c64b313.pth')

vgg16_weights = torch.load('../vgg16_bn-6c64b313.pth')
mapped_weights = {}
for k_vgg, k_segnet in zip(vgg16_weights.keys(), net.state_dict().keys()):
    if "features" in k_vgg:
        mapped_weights[k_segnet] = vgg16_weights[k_vgg]
        # print("Mapping {} to {}".format(k_vgg, k_segnet))

for it in net.state_dict().keys():

    if it == 'conv1_1_d.weight':
        avg = torch.mean(mapped_weights[it.replace('_d', '')].data, dim=1)
        mapped_weights[it] = avg.unsqueeze(1)
    if '_d' in it and it != 'conv1_1_d.weight':
        if it.replace('_d', '') in mapped_weights:
            mapped_weights[it] = mapped_weights[it.replace('_d', '')]

try:
    net.load_state_dict(mapped_weights)
    print("Loaded VGG-16 weights in SegNet !")
except:
    # Ignore missing keys
    pass

net.cuda()

# Load the datasets
train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
test_ids = ['5', '21', '15', '30']

print("Tiles for training : ", train_ids)
print("Tiles for testing : ", test_ids)

train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 0.01
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


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    net.eval()

    for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(
                tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                     leave=False)):
            # Display in progress results
            # if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
            #     _pred = np.argmax(pred, axis=-1)
            #     fig = plt.figure()
            #     fig.add_subplot(1, 3, 1)
            #     plt.imshow(np.asarray(255 * img, dtype='uint8'))
            #     fig.add_subplot(1, 3, 2)
            #     plt.imshow(convert_to_color(_pred))
            #     fig.add_subplot(1, 3, 3)
            #     plt.imshow(gt)
            #     clear_output()
            #     plt.show()

            # Build the tensor
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

            min = np.min(dsm)
            max = np.max(dsm)
            dsm = (dsm - min) / (max - min)
            dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
            dsm_patches = np.asarray(dsm_patches)
            dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)

            # Do the inference
            outs = net(image_patches, dsm_patches)
            outs = outs.data.cpu().numpy()

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out
            del (outs)

        pred = np.argmax(pred, axis=-1)

        # Display the result
        clear_output()
        # fig = plt.figure()
        # fig.add_subplot(1, 3, 1)
        # plt.imshow(np.asarray(255 * img, dtype='uint8'))
        # fig.add_subplot(1, 3, 2)
        # plt.imshow(convert_to_color(pred))
        # fig.add_subplot(1, 3, 3)
        # plt.imshow(gt)
        # plt.show()

        all_preds.append(pred)
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
    weights = weights.cuda()

    criterion = nn.NLLLoss2d(weight=weights)
    iter_ = 0
    acc_best = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
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
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
                # plt.plot(mean_losses[:iter_]) and plt.show()
                # fig = plt.figure()
                # fig.add_subplot(131)
                # plt.imshow(rgb)
                # plt.title('RGB')
                # fig.add_subplot(132)
                # plt.imshow(convert_to_color(gt))
                # plt.title('Ground truth')
                # fig.add_subplot(133)
                # plt.title('Prediction')
                # plt.imshow(convert_to_color(pred))
                # plt.show()
            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            acc = test(net, test_ids, all=False, stride=32)
            if acc > acc_best:
                torch.save(net.state_dict(), 'segnet256_epoch{}_{}'.format(e, acc))
                acc_best = acc
    print("Save final model!!")
    torch.save(net.state_dict(), 'segnet_final')

train(net, optimizer, 100, scheduler)

######   test   ####
# net.load_state_dict(torch.load('segnet256_epoch??'))
# acc, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
# print("Acc: %.2f" % (acc))
# for p, id_ in zip(all_preds, test_ids):
#    img = convert_to_color(p)
#    # plt.imshow(img) and plt.show()
#    io.imsave('inference_tile_{}.png'.format(id_), img)
