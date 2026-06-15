import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
from skimage import io
import os

# Parameters
## SwinFusion
# WINDOW_SIZE = (64, 64) # Patch size
WINDOW_SIZE = (256, 256) # Patch size

STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "/home/ubuntu22-tmp/xianping/ISPRS_dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10 # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

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

MODEL = 'singleDino'

MODE = 'Train'
# MODE = 'Test'
DATASET = 'Vaihingen'
# DATASET = 'Potsdam'
# DATASET = 'Hunan'

# print(MODEL + ', ' + MODE + ', ' + DATASET)

if DATASET == 'Vaihingen':
    train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
    test_ids = ['5', '21', '15', '30']
    Stride_Size =32
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
elif DATASET == 'Potsdam':
    train_ids = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
                '4_12', '6_8', '6_12', '6_7', '4_11']
    test_ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
    # test_ids = ['4_10', '2_11', '6_11']
    Stride_Size = 128
    MAIN_FOLDER = FOLDER + 'Potsdam/'
    DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
elif DATASET == 'Hunan':
    train_ids = ['10062', '10258', '1031', '10434', '10503', '1057', '10970', '11271', '11272', '11371', '11375',
                 '11524', '11568', '11603', '11607', '11625', '11644', '1165', '11659', '11678', '11724', '11766',
                 '11838', '11854', '11856', '11919', '11957', '11959', '12042', '12098', '12148', '12151', '12152',
                 '12154', '12241', '12244', '12350', '12363', '12563', '1262', '12669', '12728', '12813', '12863',
                 '13019', '1302', '13109', '13248', '13258', '13299', '13383', '13410', '13415', '13437', '13448',
                 '13522', '13524', '13535', '13563', '13564', '13565', '13622', '13718', '13742', '13766', '13807',
                 '1383', '13925', '13927', '13931', '13932', '13933', '13990', '13995', '14247', '14287', '14290',
                 '1436', '14463', '14467', '14477', '1450', '1451', '14603', '14694', '1471', '14755', '14758', '14763',
                 '14786', '14822', '14873', '14936', '14937', '14942', '15001', '15023', '15075', '15144', '15150',
                 '15201', '15230', '15275', '15398', '15493', '15520', '15548', '15596', '15597', '15603', '15686',
                 '15763', '15864', '1599', '15998', '16028', '16032', '1605', '16089', '16090', '16093', '1619',
                 '16217', '16356', '16516', '16530', '16541', '16582', '16699', '16702', '16703', '16709', '16742',
                 '16744', '16745', '16866', '16870', '16880', '16888', '16912', '16922', '17040', '17092', '17095',
                 '17269', '17369', '1752', '17582', '17637', '17721', '1773', '1774', '1775', '17799', '17800', '17820',
                 '17933', '17934', '17943', '18009', '18117', '18118', '18121', '18164', '18183', '18185', '18186',
                 '18298', '18371', '18732', '18950', '1899', '1906', '19098', '19149', '1928', '1932', '19609', '19680',
                 '19688', '19866', '19915', '20054', '20175', '20314', '20386', '20408', '20464', '20561', '20599',
                 '2062', '2070', '20734', '20736', '20737', '20752', '20759', '2087', '20910', '20926', '20933',
                 '20935', '2097', '2100', '21111', '21211', '21545', '21561', '21562', '21563', '21565', '21566',
                 '21592', '21601', '21610', '21736', '21737', '21738', '21740', '21747', '21820', '21922', '21938',
                 '21967', '21982', '22000', '22077', '22099', '2232', '22547', '2255', '22555', '22729', '22731',
                 '22906', '23080', '23232', '23342', '23588', '23701', '23772', '2396', '23961', '2397', '2402',
                 '24149', '2427', '2428', '2429', '24296', '2431', '24718', '24733', '25069', '2542', '25709', '25777',
                 '25778', '2584', '2597', '26032', '2604', '2606', '2617', '2618', '26622', '26628', '26776', '27163',
                 '27168', '27282', '27308', '27312', '27437', '2748', '27718', '27860', '2791', '27960', '28038',
                 '28073', '28078', '28189', '28191', '28532', '28533', '28736', '28796', '28801', '28933', '29020',
                 '29028', '29104', '29145', '29171', '29172', '29284', '29286', '29288', '29391', '29393', '29394',
                 '29567', '29571', '29621', '29638', '29733', '29779', '29796', '29829', '29844', '29886', '29892',
                 '299', '30130', '30148', '30276', '30301', '30399', '30425', '30482', '30560', '30597', '30606',
                 '30719', '31014', '31175', '31188', '3122', '31242', '31302', '31355', '31583', '31586', '3161',
                 '31621', '31708', '31797', '3817', '3936', '3994', '4171', '418', '419', '4362', '4363', '452', '4529',
                 '4530', '4887', '4889', '5043', '5066', '5221', '5223', '5232', '5236', '5238', '5412', '5830', '5994',
                 '6156', '6167', '6213', '6379', '6383', '6405', '6407', '646', '6597', '6600', '6601', '6768', '6770',
                 '6771', '6772', '6777', '6795', '7167', '726', '7329', '7330', '7331', '7421', '7473', '7898', '8232',
                 '8234', '830', '833', '8831', '8931', '8976', '9064', '9174', '940', '942', '944', '9662', '9956']
    test_ids = ['1025', '11173', '11233', '12040', '12153', '12779', '12918', '13433', '13436', '14105', '14126',
                '14155', '14169', '15184', '15695', '16214', '16577', '16724', '18079', '18895', '1930', '19471',
                '19853', '21804', '21822', '21981', '22154', '2246', '22728', '22837', '22907', '2408', '25699',
                '29654', '29675', '298', '30669', '3100', '31653', '3534', '5042', '5450', '6186', '637', '6581',
                '6794', '708', '7181', '7679', '9258', '11624', '11767', '11816', '1239', '12626', '12815', '1290',
                '1303', '13254', '13257', '13515', '13765', '14108', '14293', '15426', '1625', '16373', '16750',
                '16890', '17039', '17055', '17107', '17455', '17821', '17980', '18958', '1908', '1923', '20028', '2106',
                '21385', '21423', '22312', '2256', '2265', '2442', '2603', '27264', '28965', '29287', '30275', '3280',
                '4017', '4166', '4886', '5215', '5233', '5410', '639', '7176']
    Stride_Size = 256
    MAIN_FOLDER = "/home/ubuntu22-tmp/xianping/ISPRS_dataset/Hunan/"
    LABELS = ["cropland", "forest", "grassland", "wetland", "water", "unused land", "built-up area"]
    N_CLASSES = len(LABELS) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    DATA_FOLDER = MAIN_FOLDER + 'images_png/{}.png'
    LABEL_FOLDER = MAIN_FOLDER + 'masks_png/{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'masks_png/{}.tif'
    palette = {
        0: (196, 90, 17),       # cropland
        1: (51, 129, 88),       # forest
        2: (177, 205, 61),      # grassland
        3: (228, 84, 96),       # wetland
        4: (91, 154, 214),      # water
        5: (225, 174, 110),     # unused land
        6: (239, 159, 2)}       # built-up area
    invert_palette = {v: k for k, v in palette.items()}


print(MODEL + ', ' + MODE + ', ' + DATASET + ', WINDOW_SIZE: ', WINDOW_SIZE, 
      ', BATCH_SIZE: ' + str(BATCH_SIZE), ', Stride_Size: ', str(Stride_Size))

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

def object_process(object):
    ids = np.unique(object)
    new_id = 1
    for id in ids[1:]:
        object = np.where(object == id, new_id, object)
        new_id += 1
    return object

class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        if DATASET == 'Potsdam' or DATASET == 'Vaihingen':
            return BATCH_SIZE * 1000
        elif DATASET == 'Hunan':
            return BATCH_SIZE * 500
        else:
            return None

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
            ## Potsdam IRRG
            if DATASET == 'Potsdam':
                ## RGB
                data = io.imread(self.data_files[random_idx])[:, :, :3].transpose((2, 0, 1))
                ## IRRG
                # data = io.imread(self.data_files[random_idx])[:, :, (3, 0, 1, 2)][:, :, :3].transpose((2, 0, 1))
                data = 1 / 255 * np.asarray(data, dtype='float32')
            else:
            ## Vaihingen IRRG
                data = io.imread(self.data_files[random_idx])
                data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            if DATASET == 'Hunan':
                label = np.asarray(io.imread(self.label_files[random_idx]), dtype='int64')
            else:
                label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        if DATASET == 'Hunan':
            data_p = data
            label_p = label

        else:
            # Get a random patch
            x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
            data_p = data[:, x1:x2, y1:y2]
            label_p = label[x1:x2, y1:y2]

        # Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))


def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

class CrossEntropy2d_ignore(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
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
    
def loss_calc(pred, label, weights):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d_ignore().cuda()

    return criterion(pred, label, weights)

def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


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

class ObjectLoss(nn.Module):
  def __init__(self, max_object=50):
        super().__init__()
        self.max_object = max_object

  def forward(self, pred, gt):
    num_object = int(torch.max(gt)) + 1
    num_object = min(num_object, self.max_object)
    total_object_loss = 0

    for object_index in range(1,num_object):
        mask = torch.where(gt == object_index, 1, 0).unsqueeze(1).to('cuda')
        num_point = mask.sum(2).sum(2).unsqueeze(2).unsqueeze(2).to('cuda')
        avg_pool = mask / (num_point + 1)

        object_feature = pred.mul(avg_pool)

        avg_feature = object_feature.sum(2).sum(2).unsqueeze(2).unsqueeze(2).repeat(1,1,gt.shape[1],gt.shape[2])
        avg_feature = avg_feature.mul(mask)

        object_loss = torch.nn.functional.mse_loss(num_point * object_feature, avg_feature, reduction='mean')
        total_object_loss = total_object_loss + object_loss
      
    return total_object_loss
  
class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, _, _, _ = pred.shape
        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)
        class_map = pred.argmax(dim=1).cpu()  # Get Class Map with the Shape: [B, H, W]

        # boundary map
        gt_b = F.max_pool2d(
            1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt

        pred_b = F.max_pool2d(
            1 - class_map, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - class_map

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, 2, -1)
        pred_b = pred_b.view(n, 2, -1)
        gt_b_ext = gt_b_ext.view(n, 2, -1)
        pred_b_ext = pred_b_ext.view(n, 2, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss
    
def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

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

def metrics_hunan(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

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
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:])))
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
    MIoU = np.nanmean(MIoU[:])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")

    return MIoU