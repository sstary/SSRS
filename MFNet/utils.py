import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import itertools
from torchvision.utils import make_grid
from PIL import Image
from skimage import io
import os

# Parameters
## SwinFusion
WINDOW_SIZE = (256, 256) # Patch size

STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "/media/lscsc/nas/xianping/ISPRS_dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10
# BATCH_SIZE = 4 # For backbone ViT-Huge

MODEL = 'UNetformer'
# MODEL = 'FTUNetformer'
MODE = 'Train'
# MODE = 'Test'

FTune = 'Adapter'
# FTune = 'LoRA'

DATASET = 'Vaihingen'
# DATASET = 'Potsdam'
# DATASET = 'Hunan'
IF_SAM = True
# IF_SAM = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

# ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

if DATASET == 'Vaihingen':
    train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
    test_ids = ['5', '21', '15', '30']
    Stride_Size = 32
    epochs = 50
    save_epoch = 1
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
elif DATASET == 'Potsdam':
    train_ids = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
                '4_12', '6_8', '6_12', '6_7', '4_11']
    test_ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
    Stride_Size = 128
    epochs = 50
    save_epoch = 1
    MAIN_FOLDER = FOLDER + 'Potsdam/'
    DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
    DSM_FOLDER = MAIN_FOLDER + '1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg'
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
elif DATASET == 'Hunan':
    train_ids = ['10434' ,'11524' ,'11607' ,'11724' ,'11854' ,'11856' ,'11919' ,'12152' ,'12350' ,'12563' ,'12669' ,'12813' ,'1302' ,'13258' ,'13383' ,'13524' ,'13565' ,'13932' ,'14477' ,'14694' ,'15001' ,'15023' ,'15201' ,'15230' ,'15548' ,'15603' ,'15686' ,'1599' ,'15998' ,'16090' ,'16217' ,'16541' ,'16582' ,'16703' ,'16709' ,'17092' ,'17269' ,'18186' ,'18950' ,'1899' ,'1906' ,'19098' ,'19680' ,'19915' ,'20175' ,'20386' ,'20561' ,'20734' ,'20752' ,'20759' ,'21562' ,'21565' ,'21738' ,'21820' ,'22000' ,'2232' ,'22547' ,'22729' ,'2431' ,'24718' ,'24733' ,'25069' ,'2584' ,'2617' ,'26622' ,'27163' ,'27308' ,'27312' ,'2791' ,'29393' ,'29886' ,'30560' ,'30719' ,'31175' ,'31188' ,'31586' ,'31797' ,'3817' ,'4529' ,'4530' ,'4889' ,'5223' ,'6213' ,'6597' ,'6600' ,'6768' ,'7329' ,'8232' ,'830' ,'8931' ,'944' ,'9956' ,'2087' ,'17721' ,'13990' ,'13622' ,'13563' ,'18009' ,'12148' ,'16888' ,'14758' ,'1773' ,'16516' ,'20408' ,'2070' ,'10062' ,'17637' ,'14942' ,'13931' ,'13410' ,'11959' ,'15150' ,'17582' ,'17820' ,'21545' ,'21563' ,'21592' ,'21922' ,'2255' ,'26628' ,'28533' ,'28801' ,'29621' ,'29796' ,'30482' ,'31302' ,'31355' ,'4171' ,'4887' ,'5994' ,'6167' ,'6777' ,'7421' ,'833' ,'9064' ,'9662' ,'14936' ,'15493' ,'2097' ,'25709' ,'30301' ,'18117' ,'11766' ,'3994' ,'5830' ,'14786' ,'1774' ,'16032' ,'2597' ,'18164' ,'8976' ,'2427' ,'418' ,'23961' ,'1165' ,'6383' ,'22906' ,'26032' ,'18371' ,'6156' ,'7167' ,'20736' ,'16880' ,'29145' ,'21211' ,'7473' ,'29172' ,'22077' ,'14755' ,'2428' ,'16922' ,'15144' ,'5232' ,'25777' ,'21736' ,'14290' ,'15275' ,'1025' ,'11173' ,'12040' ,'12779' ,'14126' ,'15695' ,'16214' ,'16577' ,'18079' ,'1930' ,'21804' ,'22154' ,'25699' ,'29675' ,'298' ,'31653' ,'5042' ,'637' ,'6581' ,'708' ,'7679' ,'1031' ,'11272' ,'14463' ,'16745' ,'12244' ,'1775' ,'1752' ,'16744' ,'17095' ,'20910' ,'13742' ,'16702' ,'13925' ,'17800' ,'17040' ,'2062' ,'16912' ,'19149' ,'11371' ,'21601' ,'21610' ,'21737' ,'21747' ,'21982' ,'23232' ,'2606' ,'27860' ,'28532' ,'28933' ,'29028' ,'29284' ,'29288' ,'30597' ,'3122' ,'31242' ,'4362' ,'6405' ,'6770' ,'726' ,'7330' ,'7331' ,'8234' ,'2402' ,'646' ,'13718' ,'12363' ,'13995' ,'13807' ,'1471' ,'27168' ,'18298' ,'16093' ,'15763' ,'12042' ,'29020' ,'8831' ,'11375' ,'23772' ,'12728' ,'13448' ,'27960' ,'14467' ,'14763' ,'19866' ,'13766' ,'24296' ,'1436' ,'5236' ,'28796' ,'10258' ,'28736' ,'2100' ,'1451' ,'12918' ,'14155' ,'15184' ,'19471' ,'21822' ,'22728' ,'22837' ,'2408' ,'3100' ,'5450' ,'6186' ,'7181' ,'9258' ,'11625' ,'11644' ,'12098' ,'12154' ,'12241' ,'13248' ,'13522' ,'13564' ,'1383' ,'13927' ,'14287' ,'14822' ,'15075' ,'15520' ,'15864' ,'16028' ,'1619' ,'16699' ,'16742' ,'16866' ,'17369' ,'17934' ,'18183' ,'18185' ,'18732' ,'1928' ,'19688' ,'20314' ,'20464' ,'20737' ,'21111' ,'21561' ,'21938' ,'22555' ,'23080' ,'23588' ,'23701' ,'2604' ,'27282' ,'27718' ,'28073' ,'28189' ,'29171' ,'29286' ,'29844' ,'30148' ,'30399' ,'30425' ,'30606' ,'31014' ,'31583' ,'31621' ,'4363' ,'5043' ,'5221' ,'5238' ,'6379' ,'6407' ,'6601' ,'9174' ,'2748' ,'29829' ,'10970' ,'20926' ,'6795' ,'24149' ,'18121' ,'20935' ,'942' ,'29391' ,'29638' ,'20054' ,'3161' ,'6772' ,'17933' ,'13535' ,'5412' ,'20599' ,'299' ,'19609' ,'452' ,'28191' ,'11659' ,'1450' ,'13019' ,'11838' ,'29892' ,'12151' ,'13933' ,'11568' ,'11233' ,'12153' ,'13433' ,'13436' ,'14105' ,'14169' ,'16724' ,'18895' ,'19853' ,'21981' ,'2246' ,'22907' ,'29654' ,'30669' ,'3534' ,'6794' ,'1057' ,'11271' ,'11603' ,'11678' ,'11957' ,'1262' ,'12863' ,'13109' ,'13299' ,'13415' ,'14603' ,'14873' ,'15398' ,'15596' ,'1605' ,'16089' ,'16530' ,'16870' ,'17799' ,'17943' ,'1932' ,'21740' ,'21967' ,'23342' ,'2396' ,'2397' ,'2429' ,'2542' ,'25778' ,'2618' ,'26776' ,'28038' ,'29104' ,'29567' ,'29733' ,'29779' ,'30276' ,'31708' ,'3936' ,'419' ,'5066' ,'7898' ,'20933' ,'15597' ,'18118' ,'16356' ,'14937' ,'10503' ,'13437' ,'14247' ,'21566' ,'22099' ,'22731' ,'27437' ,'28078' ,'29394' ,'29571' ,'30130' ,'6771' ,'940']
    test_ids = ['11767' ,'11816' ,'1239' ,'12626' ,'12815' ,'1290' ,'1303' ,'13254' ,'13257' ,'13515' ,'13765' ,'14108' ,'14293' ,'15426' ,'1625' ,'16373' ,'16750' ,'16890' ,'17039' ,'17055' ,'17107' ,'17455' ,'17821' ,'17980' ,'18958' ,'1908' ,'1923' ,'20028' ,'2106' ,'21385' ,'21423' ,'22312' ,'2256' ,'2265' ,'2442' ,'2603' ,'27264' ,'28965' ,'29287' ,'30275' ,'3280' ,'4017' ,'4166' ,'4886' ,'5215' ,'5233' ,'5410' ,'639' ,'7176' ,'11624']
    Stride_Size = 256
    epochs = 50
    save_epoch = 1
    MAIN_FOLDER = "/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/"
    LABELS = ["cropland", "forest", "grassland", "wetland", "water", "unused land", "built-up area"]
    N_CLASSES = len(LABELS) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    DATA_FOLDER = MAIN_FOLDER + 'images_png/{}.png'
    DSM_FOLDER = MAIN_FOLDER + 'dsm_pngs/{}.png'
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
    
    
print(MODEL + ', ' + MODE + ', ' + DATASET + ', IF_SAM: ' + str(IF_SAM) + ', WINDOW_SIZE: ', WINDOW_SIZE, 
      ', BATCH_SIZE: ' + str(BATCH_SIZE), ', Stride_Size: ', str(Stride_Size),
      ', epochs: ' + str(epochs), ', save_epoch: ', str(save_epoch),)

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
        if DATASET == 'Potsdam' or DATASET == 'Vaihingen':
            return BATCH_SIZE * 1000
        elif  DATASET == 'Hunan':
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

        if random_idx in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[random_idx]
        else:
            # DSM is normalized in [0, 1]
            dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')
            min = np.min(dsm)
            max = np.max(dsm)
            if DATASET == 'Hunan':
                dsm = (dsm - min) / (max - min + 1e-8)
            else:
                dsm = (dsm - min) / (max - min)
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

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
            dsm_p = dsm
            label_p = label

        else:
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

def metrics_loveda(predictions, gts, label_values=LABELS):
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