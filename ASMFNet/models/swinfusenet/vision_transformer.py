# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinFuseNet(nn.Module):
    def __init__(self, img_size=224, num_classes=6, zero_head=False, vis=False):
        super(SwinFuseNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.swinfusenet = SwinTransformerSys(img_size=img_size,
                                patch_size=4,
                                in_chans=3,
                                num_classes=self.num_classes,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                depths_decoder=[ 2, 2, 2, 1],
                                window_size=7,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x, y):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        # 增加维度，在第二维增加一个维度，使其3从二维变为三维
        # print(y.shape)
        y = torch.unsqueeze(y, dim=1)
        # print(y.shape)
        y = y.repeat(1,3,1,1)
        # print(y.shape)
        logits = self.swinfusenet(x, y)
        return logits

    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swinfusenet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swinfusenet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "patch_embed" in k:
                    current_k = k.replace('embed', 'embedd')
                    full_dict.update({current_k:v})
                if "layers." in k:
                    ## DSM branch
                    current_dk = k.replace('layers', 'layersd')
                    full_dict.update({current_dk:v})
                    ## Decoder
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swinfusenet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")

