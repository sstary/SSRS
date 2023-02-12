import numpy as np
from skimage import io
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from torchvision.utils import make_grid
from PIL import Image
from torch.autograd import Variable
import os
import random

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
POWER = 0.9
def loss_calc(pred, label, weights):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d().cuda()

    return criterion(pred, label, weights)

def coral(source, target):
    source = torch.squeeze(source)
    target = torch.squeeze(target)
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, NUM_STEPS):
    lr = lr_poly(LEARNING_RATE, i_iter, NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr

def adjust_learning_rate_D(optimizer, i_iter, NUM_STEPS):
    lr = lr_poly(LEARNING_RATE_D, i_iter, NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def save_img(tensor, name):
    tensor = tensor.cpu() .permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')

WINDOW_SIZE = (256, 256)
class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids_s, ids_t, dataset_s, dataset_t, data_files_s, data_files_t, label_files_s, label_files_t,
                 cache=False, RGB_flag=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache
        self.dataset_s = dataset_s
        self.RGB_flag = RGB_flag
        self.data_files_s = [data_files_s.format(id) for id in ids_s]
        self.label_files_s = [label_files_s.format(id) for id in ids_s]

        self.dataset_t = dataset_t
        self.data_files_t = [data_files_t.format(id) for id in ids_t]
        self.label_files_t = [label_files_t.format(id) for id in ids_t]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files_s + self.label_files_s:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        for f in self.data_files_t + self.label_files_t:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))


        # Initialize cache dicts
        self.data_cache_s = {}
        self.label_cache_s = {}

        self.data_cache_t = {}
        self.label_cache_t = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        # Vaihingen
        return 10000
        # # Postam
        # return 50000

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
        random_idx_s = random.randint(0, len(self.data_files_s) - 1)
        random_idx_t = random.randint(0, len(self.data_files_t) - 1)

        # source data
        if random_idx_s in self.data_cache_s.keys():
            data_s = self.data_cache_s[random_idx_s]
        else:
            # Data is normalized in [0, 1]
            if self.dataset_s == 'Potsdam':
                if self.RGB_flag:
                    data_s = io.imread(self.data_files_s[random_idx_s])[:, :, :3].transpose((2, 0, 1))
                else:
                    data_s = io.imread(self.data_files_s[random_idx_s])[:, :, (3, 0, 1, 2)][:, :, :3].transpose((2, 0, 1))
                data_s = 1 / 255 * np.asarray(data_s, dtype='float32')
            elif self.dataset_s == 'Vaihingen':
                data_s = io.imread(self.data_files_s[random_idx_s])
                data_s = 1 / 255 * np.asarray(data_s.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_s[random_idx_s] = data_s
        if random_idx_s in self.label_cache_s.keys():
            label_s = self.label_cache_s[random_idx_s]
        else:
            # Labels are converted from RGB to their numeric values
            label_s = np.asarray(convert_from_color(io.imread(self.label_files_s[random_idx_s])), dtype='int64')
            if self.cache:
                self.label_cache_s[random_idx_s] = label_s

        # target data
        if random_idx_t in self.data_cache_t.keys():
            data_t = self.data_cache_t[random_idx_t]
        else:
            # Data is normalized in [0, 1]
            if self.dataset_t == 'Potsdam':
                if self.RGB_flag:
                    data_t = io.imread(self.data_files_t[random_idx_t])[:, :, :3].transpose((2, 0, 1))
                else:
                    data_t = io.imread(self.data_files_t[random_idx_t])[:, :, (3, 0, 1, 2)][:, :, :3].transpose((2, 0, 1))
                data_t = 1 / 255 * np.asarray(data_t, dtype='float32')
            elif self.dataset_t == 'Vaihingen':
                data_t = io.imread(self.data_files_t[random_idx_t])
                data_t = 1 / 255 * np.asarray(data_t.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_t[random_idx_t] = data_t
        if random_idx_t in self.label_cache_t.keys():
            label_t = self.label_cache_t[random_idx_t]
        else:
            # Labels are converted from RGB to their numeric values
            label_t = np.asarray(convert_from_color(io.imread(self.label_files_t[random_idx_t])), dtype='int64')
            if self.cache:
                self.label_cache_t[random_idx_t] = label_t

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data_s, WINDOW_SIZE)
        data_s = data_s[:, x1:x2, y1:y2]
        label_s = label_s[x1:x2, y1:y2]

        x1, x2, y1, y2 = get_random_pos(data_t, WINDOW_SIZE)
        data_t = data_t[:, x1:x2, y1:y2]
        label_t = label_t[x1:x2, y1:y2]

        # Data augmentation
        data_s, label_s = self.data_augmentation(data_s, label_s)
        data_t, label_t = self.data_augmentation(data_t, label_t)
        # Return the torch.Tensor values
        return (torch.from_numpy(data_s),
                torch.from_numpy(label_s),
                torch.from_numpy(data_t),
                torch.from_numpy(label_t))



def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

def BSP(feature):
    _, s_s, _ = torch.svd(feature)
    sigma = torch.pow(torch.mean(s_s), 2)
    return sigma

def discrepancy(attx, atty):
    return torch.mean(torch.abs(attx - atty))

def random_discrepancy(attx, atty):
    pos = random.randint(0, 31)
    attx = attx[:, :, pos, pos]
    atty = atty[:, :, pos, pos]
    return torch.mean(torch.abs(attx - atty))

def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-10
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def dcm(outputs_test, gent=True):
    epsilon = 1e-10
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    entropy_loss = torch.mean(entropy(softmax_out))
    if gent:
        msoftmax = softmax_out.mean(dim=0)
        msoftmax = -msoftmax * torch.log2(msoftmax + epsilon)
        gentropy_loss = torch.mean(msoftmax)
        entropy_loss -= gentropy_loss
    im_loss = entropy_loss * 1.0
    return im_loss

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" %(kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:5])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")

    return MIoU
