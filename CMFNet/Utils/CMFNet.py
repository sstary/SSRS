import copy
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import Config as config

class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Attention_org(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.query4 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(config.transformer["num_heads"]):
            query1 = nn.Linear(channel_num[0] // 4, channel_num[0] // 4, bias=False)
            query2 = nn.Linear(channel_num[1] // 4, channel_num[1] // 4, bias=False)
            query3 = nn.Linear(channel_num[2] // 4, channel_num[2] // 4, bias=False)
            query4 = nn.Linear(channel_num[3] // 4, channel_num[3] // 4, bias=False)
            key = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            value = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.query3.append(copy.deepcopy(query3))
            self.query4.append(copy.deepcopy(query4))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_Q3_list = []
        multi_head_Q4_list = []
        multi_head_K_list = []
        multi_head_V_list = []
        if emb1 is not None:
            Q0, Q1, Q2, Q3 = emb1.split(emb1.shape[2] // 4, dim=2)
            multi_head_Q1_list.append(self.query1[0](Q0))
            multi_head_Q1_list.append(self.query1[1](Q1))
            multi_head_Q1_list.append(self.query1[2](Q2))
            multi_head_Q1_list.append(self.query1[3](Q3))
        if emb2 is not None:
            Q0, Q1, Q2, Q3 = emb2.split(emb2.shape[2] // 4, dim=2)
            multi_head_Q2_list.append(self.query2[0](Q0))
            multi_head_Q2_list.append(self.query2[1](Q1))
            multi_head_Q2_list.append(self.query2[2](Q2))
            multi_head_Q2_list.append(self.query2[3](Q3))
        if emb3 is not None:
            Q0, Q1, Q2, Q3 = emb3.split(emb3.shape[2] // 4, dim=2)
            multi_head_Q3_list.append(self.query3[0](Q0))
            multi_head_Q3_list.append(self.query3[1](Q1))
            multi_head_Q3_list.append(self.query3[2](Q2))
            multi_head_Q3_list.append(self.query3[3](Q3))
        if emb4 is not None:
            Q0, Q1, Q2, Q3 = emb4.split(emb4.shape[2] // 4, dim=2)
            multi_head_Q4_list.append(self.query4[0](Q0))
            multi_head_Q4_list.append(self.query4[1](Q1))
            multi_head_Q4_list.append(self.query4[2](Q2))
            multi_head_Q4_list.append(self.query4[3](Q3))
        Q0 = torch.cat([emb_all[:, :, 0:16], emb_all[:, :, 64:96], emb_all[:, :, 192:256], emb_all[:, :, 448:576]],
                       dim=2)
        Q1 = torch.cat([emb_all[:, :, 16:32], emb_all[:, :, 96:128], emb_all[:, :, 256:320], emb_all[:, :, 576:704]],
                       dim=2)
        Q2 = torch.cat([emb_all[:, :, 32:48], emb_all[:, :, 128:160], emb_all[:, :, 320:384], emb_all[:, :, 704:832]],
                       dim=2)
        Q3 = torch.cat([emb_all[:, :, 48:64], emb_all[:, :, 160:192], emb_all[:, :, 384:448], emb_all[:, :, 832:960]],
                       dim=2)
        multi_head_K_list.append(self.key[0](Q0))
        multi_head_K_list.append(self.key[0](Q1))
        multi_head_K_list.append(self.key[0](Q2))
        multi_head_K_list.append(self.key[0](Q3))
        Q0 = torch.cat([emb_all[:, :, 0:16], emb_all[:, :, 64:96], emb_all[:, :, 192:256], emb_all[:, :, 448:576]],
                       dim=2)
        Q1 = torch.cat([emb_all[:, :, 16:32], emb_all[:, :, 96:128], emb_all[:, :, 256:320], emb_all[:, :, 576:704]],
                       dim=2)
        Q2 = torch.cat([emb_all[:, :, 32:48], emb_all[:, :, 128:160], emb_all[:, :, 320:384], emb_all[:, :, 704:832]],
                       dim=2)
        Q3 = torch.cat([emb_all[:, :, 48:64], emb_all[:, :, 160:192], emb_all[:, :, 384:448], emb_all[:, :, 832:960]],
                       dim=2)
        multi_head_V_list.append(self.value[0](Q0))
        multi_head_V_list.append(self.value[0](Q1))
        multi_head_V_list.append(self.value[0](Q2))
        multi_head_V_list.append(self.value[0](Q3))

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1) if emb1 is not None else None
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1) if emb2 is not None else None
        multi_head_Q3 = torch.stack(multi_head_Q3_list, dim=1) if emb3 is not None else None
        multi_head_Q4 = torch.stack(multi_head_Q4_list, dim=1) if emb4 is not None else None
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2) if emb1 is not None else None
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None
        multi_head_Q3 = multi_head_Q3.transpose(-1, -2) if emb3 is not None else None
        multi_head_Q4 = multi_head_Q4.transpose(-1, -2) if emb4 is not None else None

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K) if emb1 is not None else None
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K) if emb2 is not None else None
        attention_scores3 = torch.matmul(multi_head_Q3, multi_head_K) if emb3 is not None else None
        attention_scores4 = torch.matmul(multi_head_Q4, multi_head_K) if emb4 is not None else None

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size) if emb1 is not None else None
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size) if emb2 is not None else None
        attention_scores3 = attention_scores3 / math.sqrt(self.KV_size) if emb3 is not None else None
        attention_scores4 = attention_scores4 / math.sqrt(self.KV_size) if emb4 is not None else None

        attention_probs1 = self.softmax(self.psi(attention_scores1)) if emb1 is not None else None
        attention_probs2 = self.softmax(self.psi(attention_scores2)) if emb2 is not None else None
        attention_probs3 = self.softmax(self.psi(attention_scores3)) if emb3 is not None else None
        attention_probs4 = self.softmax(self.psi(attention_scores4)) if emb4 is not None else None
        # print(attention_probs4.size())

        if self.vis:
            weights = []
            weights.append(attention_probs1.mean(1))
            weights.append(attention_probs2.mean(1))
            weights.append(attention_probs3.mean(1))
            weights.append(attention_probs4.mean(1))
        else:
            weights = None

        attention_probs1 = self.attn_dropout(attention_probs1) if emb1 is not None else None
        attention_probs2 = self.attn_dropout(attention_probs2) if emb2 is not None else None
        attention_probs3 = self.attn_dropout(attention_probs3) if emb3 is not None else None
        attention_probs4 = self.attn_dropout(attention_probs4) if emb4 is not None else None

        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V) if emb1 is not None else None
        context_layer2 = torch.matmul(attention_probs2, multi_head_V) if emb2 is not None else None
        context_layer3 = torch.matmul(attention_probs3, multi_head_V) if emb3 is not None else None
        context_layer4 = torch.matmul(attention_probs4, multi_head_V) if emb4 is not None else None

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous() if emb1 is not None else None
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous() if emb2 is not None else None
        context_layer3 = context_layer3.permute(0, 3, 2, 1).contiguous() if emb3 is not None else None
        context_layer4 = context_layer4.permute(0, 3, 2, 1).contiguous() if emb4 is not None else None
        context_layer1 = context_layer1.view(context_layer1.shape[0], context_layer1.shape[1],
                                             context_layer1.shape[2] * 4)
        context_layer2 = context_layer2.view(context_layer2.shape[0], context_layer2.shape[1],
                                             context_layer2.shape[2] * 4)
        context_layer3 = context_layer3.view(context_layer3.shape[0], context_layer3.shape[1],
                                             context_layer3.shape[2] * 4)
        context_layer4 = context_layer4.view(context_layer4.shape[0], context_layer4.shape[1],
                                             context_layer4.shape[2] * 4)

        O1 = self.out1(context_layer1) if emb1 is not None else None
        O2 = self.out2(context_layer2) if emb2 is not None else None
        O3 = self.out3(context_layer3) if emb3 is not None else None
        O4 = self.out4(context_layer4) if emb4 is not None else None
        O1 = self.proj_dropout(O1) if emb1 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None
        O3 = self.proj_dropout(O3) if emb3 is not None else None
        O4 = self.proj_dropout(O4) if emb4 is not None else None
        return O1, O2, O3, O4, weights


class Attention_org_cross(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_org_cross, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.query4 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        self.queryd1 = nn.ModuleList()
        self.queryd2 = nn.ModuleList()
        self.queryd3 = nn.ModuleList()
        self.queryd4 = nn.ModuleList()
        self.keyd = nn.ModuleList()
        self.valued = nn.ModuleList()

        for _ in range(config.transformer["num_heads"]):
            query1 = nn.Linear(channel_num[0] // 4, channel_num[0] // 4, bias=False)
            query2 = nn.Linear(channel_num[1] // 4, channel_num[1] // 4, bias=False)
            query3 = nn.Linear(channel_num[2] // 4, channel_num[2] // 4, bias=False)
            query4 = nn.Linear(channel_num[3] // 4, channel_num[3] // 4, bias=False)
            key = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            value = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.query3.append(copy.deepcopy(query3))
            self.query4.append(copy.deepcopy(query4))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))

            queryd1 = nn.Linear(channel_num[0] // 4, channel_num[0] // 4, bias=False)
            queryd2 = nn.Linear(channel_num[1] // 4, channel_num[1] // 4, bias=False)
            queryd3 = nn.Linear(channel_num[2] // 4, channel_num[2] // 4, bias=False)
            queryd4 = nn.Linear(channel_num[3] // 4, channel_num[3] // 4, bias=False)
            keyd = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            valued = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            self.queryd1.append(copy.deepcopy(queryd1))
            self.queryd2.append(copy.deepcopy(queryd2))
            self.queryd3.append(copy.deepcopy(queryd3))
            self.queryd4.append(copy.deepcopy(queryd4))
            self.keyd.append(copy.deepcopy(keyd))
            self.valued.append(copy.deepcopy(valued))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.psid = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
        self.outd1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.outd2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.outd3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.outd4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, emb1, emb2, emb3, emb4, emb_all, embd1, embd2, embd3, embd4, emb_alld):
        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_Q3_list = []
        multi_head_Q4_list = []
        multi_head_K_list = []
        multi_head_V_list = []

        multi_head_Qd1_list = []
        multi_head_Qd2_list = []
        multi_head_Qd3_list = []
        multi_head_Qd4_list = []
        multi_head_Kd_list = []
        multi_head_Vd_list = []

        if emb1 is not None:
            Q0, Q1, Q2, Q3 = emb1.split(emb1.shape[2] // 4, dim=2)
            multi_head_Q1_list.append(self.query1[0](Q0))
            multi_head_Q1_list.append(self.query1[1](Q1))
            multi_head_Q1_list.append(self.query1[2](Q2))
            multi_head_Q1_list.append(self.query1[3](Q3))
        if emb2 is not None:
            Q0, Q1, Q2, Q3 = emb2.split(emb2.shape[2] // 4, dim=2)
            multi_head_Q2_list.append(self.query2[0](Q0))
            multi_head_Q2_list.append(self.query2[1](Q1))
            multi_head_Q2_list.append(self.query2[2](Q2))
            multi_head_Q2_list.append(self.query2[3](Q3))
        if emb3 is not None:
            Q0, Q1, Q2, Q3 = emb3.split(emb3.shape[2] // 4, dim=2)
            multi_head_Q3_list.append(self.query3[0](Q0))
            multi_head_Q3_list.append(self.query3[1](Q1))
            multi_head_Q3_list.append(self.query3[2](Q2))
            multi_head_Q3_list.append(self.query3[3](Q3))
        if emb4 is not None:
            Q0, Q1, Q2, Q3 = emb4.split(emb4.shape[2] // 4, dim=2)
            multi_head_Q4_list.append(self.query4[0](Q0))
            multi_head_Q4_list.append(self.query4[1](Q1))
            multi_head_Q4_list.append(self.query4[2](Q2))
            multi_head_Q4_list.append(self.query4[3](Q3))
        # Q0, Q1, Q2, Q3 = emb_all.split(emb_all.shape[2] // 4, dim=2)
        Q0 = torch.cat([emb_all[:, :, 0:16], emb_all[:, :, 64:96], emb_all[:, :, 192:256], emb_all[:, :, 448:576]],
                       dim=2)
        Q1 = torch.cat([emb_all[:, :, 16:32], emb_all[:, :, 96:128], emb_all[:, :, 256:320], emb_all[:, :, 576:704]],
                       dim=2)
        Q2 = torch.cat([emb_all[:, :, 32:48], emb_all[:, :, 128:160], emb_all[:, :, 320:384], emb_all[:, :, 704:832]],
                       dim=2)
        Q3 = torch.cat([emb_all[:, :, 48:64], emb_all[:, :, 160:192], emb_all[:, :, 384:448], emb_all[:, :, 832:960]],
                       dim=2)
        multi_head_K_list.append(self.key[0](Q0))
        multi_head_K_list.append(self.key[0](Q1))
        multi_head_K_list.append(self.key[0](Q2))
        multi_head_K_list.append(self.key[0](Q3))
        # Q0, Q1, Q2, Q3 = emb_all.split(emb_all.shape[2] // 4, dim=2)
        Q0 = torch.cat([emb_all[:, :, 0:16], emb_all[:, :, 64:96], emb_all[:, :, 192:256], emb_all[:, :, 448:576]],
                       dim=2)
        Q1 = torch.cat([emb_all[:, :, 16:32], emb_all[:, :, 96:128], emb_all[:, :, 256:320], emb_all[:, :, 576:704]],
                       dim=2)
        Q2 = torch.cat([emb_all[:, :, 32:48], emb_all[:, :, 128:160], emb_all[:, :, 320:384], emb_all[:, :, 704:832]],
                       dim=2)
        Q3 = torch.cat([emb_all[:, :, 48:64], emb_all[:, :, 160:192], emb_all[:, :, 384:448], emb_all[:, :, 832:960]],
                       dim=2)
        multi_head_V_list.append(self.value[0](Q0))
        multi_head_V_list.append(self.value[0](Q1))
        multi_head_V_list.append(self.value[0](Q2))
        multi_head_V_list.append(self.value[0](Q3))

        if embd1 is not None:
            Q0, Q1, Q2, Q3 = embd1.split(embd1.shape[2] // 4, dim=2)
            multi_head_Qd1_list.append(self.queryd1[0](Q0))
            multi_head_Qd1_list.append(self.queryd1[1](Q1))
            multi_head_Qd1_list.append(self.queryd1[2](Q2))
            multi_head_Qd1_list.append(self.queryd1[3](Q3))
        if embd2 is not None:
            Q0, Q1, Q2, Q3 = embd2.split(embd2.shape[2] // 4, dim=2)
            multi_head_Qd2_list.append(self.queryd2[0](Q0))
            multi_head_Qd2_list.append(self.queryd2[1](Q1))
            multi_head_Qd2_list.append(self.queryd2[2](Q2))
            multi_head_Qd2_list.append(self.queryd2[3](Q3))
        if embd3 is not None:
            Q0, Q1, Q2, Q3 = embd3.split(embd3.shape[2] // 4, dim=2)
            multi_head_Qd3_list.append(self.queryd3[0](Q0))
            multi_head_Qd3_list.append(self.queryd3[1](Q1))
            multi_head_Qd3_list.append(self.queryd3[2](Q2))
            multi_head_Qd3_list.append(self.queryd3[3](Q3))
        if embd4 is not None:
            Q0, Q1, Q2, Q3 = embd4.split(embd4.shape[2] // 4, dim=2)
            multi_head_Qd4_list.append(self.queryd4[0](Q0))
            multi_head_Qd4_list.append(self.queryd4[1](Q1))
            multi_head_Qd4_list.append(self.queryd4[2](Q2))
            multi_head_Qd4_list.append(self.queryd4[3](Q3))
        Q0 = torch.cat([emb_alld[:, :, 0:16], emb_alld[:, :, 64:96], emb_alld[:, :, 192:256], emb_alld[:, :, 448:576]],
                       dim=2)
        Q1 = torch.cat(
            [emb_alld[:, :, 16:32], emb_alld[:, :, 96:128], emb_alld[:, :, 256:320], emb_alld[:, :, 576:704]], dim=2)
        Q2 = torch.cat(
            [emb_alld[:, :, 32:48], emb_alld[:, :, 128:160], emb_alld[:, :, 320:384], emb_alld[:, :, 704:832]], dim=2)
        Q3 = torch.cat(
            [emb_alld[:, :, 48:64], emb_alld[:, :, 160:192], emb_alld[:, :, 384:448], emb_alld[:, :, 832:960]], dim=2)
        multi_head_Kd_list.append(self.keyd[0](Q0))
        multi_head_Kd_list.append(self.keyd[0](Q1))
        multi_head_Kd_list.append(self.keyd[0](Q2))
        multi_head_Kd_list.append(self.keyd[0](Q3))
        Q0 = torch.cat([emb_alld[:, :, 0:16], emb_alld[:, :, 64:96], emb_alld[:, :, 192:256], emb_alld[:, :, 448:576]],
                       dim=2)
        Q1 = torch.cat(
            [emb_alld[:, :, 16:32], emb_alld[:, :, 96:128], emb_alld[:, :, 256:320], emb_alld[:, :, 576:704]], dim=2)
        Q2 = torch.cat(
            [emb_alld[:, :, 32:48], emb_alld[:, :, 128:160], emb_alld[:, :, 320:384], emb_alld[:, :, 704:832]], dim=2)
        Q3 = torch.cat(
            [emb_alld[:, :, 48:64], emb_alld[:, :, 160:192], emb_alld[:, :, 384:448], emb_alld[:, :, 832:960]], dim=2)
        multi_head_Vd_list.append(self.valued[0](Q0))
        multi_head_Vd_list.append(self.valued[0](Q1))
        multi_head_Vd_list.append(self.valued[0](Q2))
        multi_head_Vd_list.append(self.valued[0](Q3))

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1)
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1)
        multi_head_Q3 = torch.stack(multi_head_Q3_list, dim=1)
        multi_head_Q4 = torch.stack(multi_head_Q4_list, dim=1)
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        multi_head_Qd1 = torch.stack(multi_head_Qd1_list, dim=1)
        multi_head_Qd2 = torch.stack(multi_head_Qd2_list, dim=1)
        multi_head_Qd3 = torch.stack(multi_head_Qd3_list, dim=1)
        multi_head_Qd4 = torch.stack(multi_head_Qd4_list, dim=1)
        multi_head_Kd = torch.stack(multi_head_Kd_list, dim=1)
        multi_head_Vd = torch.stack(multi_head_Vd_list, dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2)
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2)
        multi_head_Q3 = multi_head_Q3.transpose(-1, -2)
        multi_head_Q4 = multi_head_Q4.transpose(-1, -2)

        multi_head_Qd1 = multi_head_Qd1.transpose(-1, -2)
        multi_head_Qd2 = multi_head_Qd2.transpose(-1, -2)
        multi_head_Qd3 = multi_head_Qd3.transpose(-1, -2)
        multi_head_Qd4 = multi_head_Qd4.transpose(-1, -2)

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_Kd)
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_Kd)
        attention_scores3 = torch.matmul(multi_head_Q3, multi_head_Kd)
        attention_scores4 = torch.matmul(multi_head_Q4, multi_head_Kd)

        attention_scoresd1 = torch.matmul(multi_head_Qd1, multi_head_K)
        attention_scoresd2 = torch.matmul(multi_head_Qd2, multi_head_K)
        attention_scoresd3 = torch.matmul(multi_head_Qd3, multi_head_K)
        attention_scoresd4 = torch.matmul(multi_head_Qd4, multi_head_K)

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size)
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size)
        attention_scores3 = attention_scores3 / math.sqrt(self.KV_size)
        attention_scores4 = attention_scores4 / math.sqrt(self.KV_size)

        attention_scoresd1 = attention_scoresd1 / math.sqrt(self.KV_size)
        attention_scoresd2 = attention_scoresd2 / math.sqrt(self.KV_size)
        attention_scoresd3 = attention_scoresd3 / math.sqrt(self.KV_size)
        attention_scoresd4 = attention_scoresd4 / math.sqrt(self.KV_size)

        attention_probs1 = self.softmax(self.psi(attention_scores1))
        attention_probs2 = self.softmax(self.psi(attention_scores2))
        attention_probs3 = self.softmax(self.psi(attention_scores3))
        attention_probs4 = self.softmax(self.psi(attention_scores4))

        attention_probsd1 = self.softmax(self.psid(attention_scoresd1))
        attention_probsd2 = self.softmax(self.psid(attention_scoresd2))
        attention_probsd3 = self.softmax(self.psid(attention_scoresd3))
        attention_probsd4 = self.softmax(self.psid(attention_scoresd4))
        # print(attention_probs4.size())

        if self.vis:
            weights = []
            weights.append(attention_probs1.mean(1))
            weights.append(attention_probs2.mean(1))
            weights.append(attention_probs3.mean(1))
            weights.append(attention_probs4.mean(1))
        else:
            weights = None

        attention_probs1 = self.attn_dropout(attention_probs1)
        attention_probs2 = self.attn_dropout(attention_probs2)
        attention_probs3 = self.attn_dropout(attention_probs3)
        attention_probs4 = self.attn_dropout(attention_probs4)

        attention_probsd1 = self.attn_dropout(attention_probsd1)
        attention_probsd2 = self.attn_dropout(attention_probsd2)
        attention_probsd3 = self.attn_dropout(attention_probsd3)
        attention_probsd4 = self.attn_dropout(attention_probsd4)

        multi_head_V = multi_head_V.transpose(-1, -2)
        multi_head_Vd = multi_head_Vd.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V)
        context_layer2 = torch.matmul(attention_probs2, multi_head_V)
        context_layer3 = torch.matmul(attention_probs3, multi_head_V)
        context_layer4 = torch.matmul(attention_probs4, multi_head_V)

        context_layerd1 = torch.matmul(attention_probsd1, multi_head_Vd)
        context_layerd2 = torch.matmul(attention_probsd2, multi_head_Vd)
        context_layerd3 = torch.matmul(attention_probsd3, multi_head_Vd)
        context_layerd4 = torch.matmul(attention_probsd4, multi_head_Vd)

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous()
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous()
        context_layer3 = context_layer3.permute(0, 3, 2, 1).contiguous()
        context_layer4 = context_layer4.permute(0, 3, 2, 1).contiguous()

        context_layerd1 = context_layerd1.permute(0, 3, 2, 1).contiguous()
        context_layerd2 = context_layerd2.permute(0, 3, 2, 1).contiguous()
        context_layerd3 = context_layerd3.permute(0, 3, 2, 1).contiguous()
        context_layerd4 = context_layerd4.permute(0, 3, 2, 1).contiguous()
        # context_layer1 = context_layer1.mean(dim=3)
        # context_layer2 = context_layer2.mean(dim=3)
        # context_layer3 = context_layer3.mean(dim=3)
        # context_layer4 = context_layer4.mean(dim=3)
        # context_layerd1 = context_layerd1.mean(dim=3)
        # context_layerd2 = context_layerd2.mean(dim=3)
        # context_layerd3 = context_layerd3.mean(dim=3)
        # context_layerd4 = context_layerd4.mean(dim=3)
        context_layer1 = context_layer1.view(context_layer1.shape[0], context_layer1.shape[1],
                                             context_layer1.shape[2] * 4)
        context_layer2 = context_layer2.view(context_layer2.shape[0], context_layer2.shape[1],
                                             context_layer2.shape[2] * 4)
        context_layer3 = context_layer3.view(context_layer3.shape[0], context_layer3.shape[1],
                                             context_layer3.shape[2] * 4)
        context_layer4 = context_layer4.view(context_layer4.shape[0], context_layer4.shape[1],
                                             context_layer4.shape[2] * 4)
        context_layerd1 = context_layerd1.view(context_layerd1.shape[0], context_layerd1.shape[1],
                                               context_layerd1.shape[2] * 4)
        context_layerd2 = context_layerd2.view(context_layerd2.shape[0], context_layerd2.shape[1],
                                               context_layerd2.shape[2] * 4)
        context_layerd3 = context_layerd3.view(context_layerd3.shape[0], context_layerd3.shape[1],
                                               context_layerd3.shape[2] * 4)
        context_layerd4 = context_layerd4.view(context_layerd4.shape[0], context_layerd4.shape[1],
                                               context_layerd4.shape[2] * 4)

        O1 = self.out1(context_layer1)
        O2 = self.out2(context_layer2)
        O3 = self.out3(context_layer3)
        O4 = self.out4(context_layer4)
        Od1 = self.outd1(context_layerd1)
        Od2 = self.outd2(context_layerd2)
        Od3 = self.outd3(context_layerd3)
        Od4 = self.outd4(context_layerd4)
        O1 = self.proj_dropout(O1)
        O2 = self.proj_dropout(O2)
        O3 = self.proj_dropout(O3)
        O4 = self.proj_dropout(O4)
        Od1 = self.proj_dropout(Od1)
        Od2 = self.proj_dropout(Od2)
        Od3 = self.proj_dropout(Od3)
        Od4 = self.proj_dropout(Od4)
        return O1, O2, O3, O4, Od1, Od2, Od3, Od4, weights


class Mlp(nn.Module):
    def __init__(self, config, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.attn_norm = LayerNorm(config.KV_size, eps=1e-6)
        self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.ffn1 = Mlp(config, channel_num[0], channel_num[0] * expand_ratio)
        self.ffn2 = Mlp(config, channel_num[1], channel_num[1] * expand_ratio)
        self.ffn3 = Mlp(config, channel_num[2], channel_num[2] * expand_ratio)
        self.ffn4 = Mlp(config, channel_num[3], channel_num[3] * expand_ratio)

    def forward(self, emb1, emb2, emb3, emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat, dim=2)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, weights = layer_block(emb1, emb2, emb3, emb4)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1, emb2, emb3, emb4, attn_weights


class ChannelTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2,
                                               in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, self.patchSize_3, img_size=img_size // 4,
                                               in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config, self.patchSize_4, img_size=img_size // 8,
                                               in_channels=channel_num[3])
        self.encoder = Encoder(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,
                                         scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,
                                         scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,
                                         scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,
                                         scale_factor=(self.patchSize_4, self.patchSize_4))

    def forward(self, en1, en2, en3, en4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(emb1, emb2, emb3,
                                                                            emb4)  # (B, n_patch, hidden)
        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None

        x1 = x1 + en1 if en1 is not None else None
        x2 = x2 + en2 if en2 is not None else None
        x3 = x3 + en3 if en3 is not None else None
        x4 = x4 + en4 if en4 is not None else None

        return x1, x2, x3, x4, attn_weights


class Block_ViT_cross(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT_cross, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3], eps=1e-6)

        self.attn_normd1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_normd2 = LayerNorm(channel_num[1], eps=1e-6)
        self.attn_normd3 = LayerNorm(channel_num[2], eps=1e-6)
        self.attn_normd4 = LayerNorm(channel_num[3], eps=1e-6)

        self.attn_normx = LayerNorm(config.KV_size, eps=1e-6)
        self.attn_normy = LayerNorm(config.KV_size, eps=1e-6)
        self.channel_attn = Attention_org_cross(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.ffn_normd1 = LayerNorm(channel_num[0], eps=1e-6)
        self.ffn_normd2 = LayerNorm(channel_num[1], eps=1e-6)
        self.ffn_normd3 = LayerNorm(channel_num[2], eps=1e-6)
        self.ffn_normd4 = LayerNorm(channel_num[3], eps=1e-6)

        self.ffn1 = Mlp(config, channel_num[0], channel_num[0] * expand_ratio)
        self.ffn2 = Mlp(config, channel_num[1], channel_num[1] * expand_ratio)
        self.ffn3 = Mlp(config, channel_num[2], channel_num[2] * expand_ratio)
        self.ffn4 = Mlp(config, channel_num[3], channel_num[3] * expand_ratio)
        self.ffnd1 = Mlp(config, channel_num[0], channel_num[0] * expand_ratio)
        self.ffnd2 = Mlp(config, channel_num[1], channel_num[1] * expand_ratio)
        self.ffnd3 = Mlp(config, channel_num[2], channel_num[2] * expand_ratio)
        self.ffnd4 = Mlp(config, channel_num[3], channel_num[3] * expand_ratio)

    def forward(self, emb1, emb2, emb3, emb4, embd1, embd2, embd3, embd4):
        embcat = []
        embcatd = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        orgd1 = embd1
        orgd2 = embd2
        orgd3 = embd3
        orgd4 = embd4
        for i in range(4):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

            var_name = "embd" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcatd.append(tmp_var)

        emb_all = torch.cat(embcat, dim=2)
        emb_alld = torch.cat(embcatd, dim=2)
        cx1 = self.attn_norm1(emb1)
        cx2 = self.attn_norm2(emb2)
        cx3 = self.attn_norm3(emb3)
        cx4 = self.attn_norm4(emb4)

        cy1 = self.attn_normd1(embd1)
        cy2 = self.attn_normd2(embd2)
        cy3 = self.attn_normd3(embd3)
        cy4 = self.attn_normd4(embd4)

        emb_all = self.attn_normx(emb_all)
        emb_alld = self.attn_normy(emb_alld)
        cx1, cx2, cx3, cx4, cy1, cy2, cy3, cy4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all, cy1, cy2, cy3,
                                                                            cy4, emb_alld)
        cx1 = org1 + cx1
        cx2 = org2 + cx2
        cx3 = org3 + cx3
        cx4 = org4 + cx4

        cy1 = orgd1 + cy1
        cy2 = orgd2 + cy2
        cy3 = orgd3 + cy3
        cy4 = orgd4 + cy4

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4

        orgd1 = cy1
        orgd2 = cy2
        orgd3 = cy3
        orgd4 = cy4
        x1 = self.ffn_norm1(cx1)
        x2 = self.ffn_norm2(cx2)
        x3 = self.ffn_norm3(cx3)
        x4 = self.ffn_norm4(cx4)

        y1 = self.ffn_normd1(cy1)
        y2 = self.ffn_normd2(cy2)
        y3 = self.ffn_normd3(cy3)
        y4 = self.ffn_normd4(cy4)
        x1 = self.ffn1(x1)
        x2 = self.ffn2(x2)
        x3 = self.ffn3(x3)
        x4 = self.ffn4(x4)

        y1 = self.ffnd1(y1)
        y2 = self.ffnd2(y2)
        y3 = self.ffnd3(y3)
        y4 = self.ffnd4(y4)
        x1 = x1 + org1
        x2 = x2 + org2
        x3 = x3 + org3
        x4 = x4 + org4

        y1 = y1 + orgd1
        y2 = y2 + orgd2
        y3 = y3 + orgd3
        y4 = y4 + orgd4

        return x1, x2, x3, x4, y1, y2, y3, y4, weights


class Encoder_cross(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder_cross, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT_cross(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, emb4, embd1, embd2, embd3, embd4):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, embd1, embd2, embd3, embd4, weights = layer_block(emb1, emb2, emb3, emb4, embd1,
                                                                                      embd2, embd3, embd4)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1)
        emb2 = self.encoder_norm2(emb2)
        emb3 = self.encoder_norm3(emb3)
        emb4 = self.encoder_norm4(emb4)
        embd1 = self.encoder_norm1(embd1)
        embd2 = self.encoder_norm2(embd2)
        embd3 = self.encoder_norm3(embd3)
        embd4 = self.encoder_norm4(embd4)
        return emb1, emb2, emb3, emb4, embd1, embd2, embd3, embd4, attn_weights


class ChannelTransformer_cross(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2,
                                               in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, self.patchSize_3, img_size=img_size // 4,
                                               in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config, self.patchSize_4, img_size=img_size // 8,
                                               in_channels=channel_num[3])

        self.embeddingsd_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddingsd_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2,
                                                in_channels=channel_num[1])
        self.embeddingsd_3 = Channel_Embeddings(config, self.patchSize_3, img_size=img_size // 4,
                                                in_channels=channel_num[2])
        self.embeddingsd_4 = Channel_Embeddings(config, self.patchSize_4, img_size=img_size // 8,
                                                in_channels=channel_num[3])
        self.encoder = Encoder_cross(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,
                                         scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,
                                         scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,
                                         scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,
                                         scale_factor=(self.patchSize_4, self.patchSize_4))
        self.reconstruct_d1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,
                                          scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_d2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,
                                          scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_d3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,
                                          scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_d4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,
                                          scale_factor=(self.patchSize_4, self.patchSize_4))

    def forward(self, en1, en2, en3, en4, end1, end2, end3, end4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        embd1 = self.embeddingsd_1(end1)
        embd2 = self.embeddingsd_2(end2)
        embd3 = self.embeddingsd_3(end3)
        embd4 = self.embeddingsd_4(end4)

        encoded1, encoded2, encoded3, encoded4, encodedd1, encodedd2, encodedd3, encodedd4, attn_weights = self.encoder(
            emb1, emb2, emb3, emb4, embd1,
            embd2, embd3,
            embd4)  # (B, n_patch, hidden)
        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None
        y1 = self.reconstruct_d1(encodedd1) if en1 is not None else None
        y2 = self.reconstruct_d2(encodedd2) if en2 is not None else None
        y3 = self.reconstruct_d3(encodedd3) if en3 is not None else None
        y4 = self.reconstruct_d4(encodedd4) if end4 is not None else None

        x1 = x1 + en1 if en1 is not None else None
        x2 = x2 + en2 if en2 is not None else None
        x3 = x3 + en3 if en3 is not None else None
        x4 = x4 + en4 if en4 is not None else None
        y1 = y1 + end1 if end1 is not None else None
        y2 = y2 + end2 if end2 is not None else None
        y3 = y3 + end3 if end3 is not None else None
        y4 = y4 + end4 if end4 is not None else None

        return x1, x2, x3, x4, y1, y2, y3, y4, attn_weights

class CMFNet(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()

        return nn.Sequential(pool_attention, conv_attention, activate)

    def __init__(self, in_channels=3, out_channels=6):
        super(CMFNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        ##### RGB ENCODER ####
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)

        ##### DSM ENCODER ####
        self.conv1_1_d = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1_1_d_bn = nn.BatchNorm2d(64)
        self.conv1_2_d = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_d_bn = nn.BatchNorm2d(64)

        self.conv2_1_d = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_d_bn = nn.BatchNorm2d(128)
        self.conv2_2_d = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_d_bn = nn.BatchNorm2d(128)

        self.conv3_1_d = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_d_bn = nn.BatchNorm2d(256)
        self.conv3_2_d = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_d_bn = nn.BatchNorm2d(256)
        self.conv3_3_d = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_d_bn = nn.BatchNorm2d(256)

        self.conv4_1_d = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_d_bn = nn.BatchNorm2d(512)
        self.conv4_2_d = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_d_bn = nn.BatchNorm2d(512)
        self.conv4_3_d = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_d_bn = nn.BatchNorm2d(512)

        self.conv5_1_d = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_d_bn = nn.BatchNorm2d(512)
        self.conv5_2_d = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_d_bn = nn.BatchNorm2d(512)
        self.conv5_3_d = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_d_bn = nn.BatchNorm2d(512)

        ##### FUSION MODULE ####
        self.attention_1 = self.attention(64)
        self.attention_2 = self.attention(128)
        self.attention_3 = self.attention(256)
        self.attention_4 = self.attention(512)
        self.attention_5 = self.attention(512)
        self.attention_1_d = self.attention(64)
        self.attention_2_d = self.attention(128)
        self.attention_3_d = self.attention(256)
        self.attention_4_d = self.attention(512)
        self.attention_5_d = self.attention(512)

        ##### SKIP MODULE: UCTransNet ####
        vis = True
        config_vit = config.get_CTranS_config()
        self.mtc = ChannelTransformer_cross(config_vit, vis, 256,
                                      channel_num=[64, 128, 256, 512],
                                      patchSize=config_vit.patch_sizes)
        self.mtc1 = ChannelTransformer(config_vit, vis, 256,
                                      channel_num=[64, 128, 256, 512],
                                      patchSize=config_vit.patch_sizes)
        ####  RGB DECODER  ####
        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)

        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)

        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)

        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)

        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)

        self.apply(self.weight_init)

    def forward(self, x, y):
        ########  DEPTH ENCODER  ########
        # Encoder block 1
        y = y.unsqueeze(1)
        y = self.conv1_1_d_bn(F.relu(self.conv1_1_d(y)))
        y1 = self.conv1_2_d_bn(F.relu(self.conv1_2_d(y)))
        y, mask1_d = self.pool(y1)

        # Encoder block 2
        y = self.conv2_1_d_bn(F.relu(self.conv2_1_d(y)))
        y2 = self.conv2_2_d_bn(F.relu(self.conv2_2_d(y)))
        y, mask2_d = self.pool(y2)

        # Encoder block 3
        y = self.conv3_1_d_bn(F.relu(self.conv3_1_d(y)))
        y = self.conv3_2_d_bn(F.relu(self.conv3_2_d(y)))
        y3 = self.conv3_3_d_bn(F.relu(self.conv3_3_d(y)))
        y, mask3_d = self.pool(y3)

        # Encoder block 4
        y = self.conv4_1_d_bn(F.relu(self.conv4_1_d(y)))
        y = self.conv4_2_d_bn(F.relu(self.conv4_2_d(y)))
        y4 = self.conv4_3_d_bn(F.relu(self.conv4_3_d(y)))
        # y, mask4_d = self.pool(y4)

        ########  RGB ENCODER  ########
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1 = x
        x, mask1 = self.pool(x1)

        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x2 = x
        x, mask2 = self.pool(x2)

        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x3 = x
        x, mask3 = self.pool(x3)

        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x4 = x

        #### Serial mode: x1-x4 from SE fusion models
        xtf1, xtf2, xtf3, xtf4, ytf1, ytf2, ytf3, ytf4, att_weights = self.mtc(x1, x2, x3, x4, y1, y2, y3, y4)
        xtf1, xtf2, xtf3, xtf4, att_weights = self.mtc1(xtf1, xtf2, xtf3, xtf4)
        x, mask4 = self.pool(x4)
        y, mask4_d = self.pool(y4)

        # Encoder block y5
        y = self.conv5_1_d_bn(F.relu(self.conv5_1_d(y)))
        y = self.conv5_2_d_bn(F.relu(self.conv5_2_d(y)))
        y5 = self.conv5_3_d_bn(F.relu(self.conv5_3_d(y)))

        # Encoder block x5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x_attention = self.attention_5(x)
        y_attention = self.attention_5_d(y5)
        x = torch.mul(x, x_attention)
        y = torch.mul(y5, y_attention)
        x5 = x + y
        x, mask5 = self.pool(x5)

        ########  DECODER  ########
        # Decoder block 5
        x = self.unpool(x, mask5)
        x = x + x5
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 4
        x = self.unpool(x, mask4)
        x = x + x4 + xtf4
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool(x, mask3)
        x = x + x3 + xtf3
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool(x, mask2)
        x = x + x2 + xtf2
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool(x, mask1)
        x = x + x1 + xtf1
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = F.log_softmax(self.conv1_1_D(x))
        return x
