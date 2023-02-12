import torch.nn as nn
import numpy as np
import torch
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import copy
import math

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
        out_channels = out_channels
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

# class Attention_org_cross(nn.Module):
#     def __init__(self, config, vis, channel_num):
#         super(Attention_org_cross, self).__init__()
#         self.vis = vis
#         self.KV_size = config.KV_sizec
#         self.channel_num = channel_num
#         self.num_attention_heads = config.transformer["num_heads"]

#         self.query = nn.ModuleList()
#         self.key = nn.ModuleList()
#         self.value = nn.ModuleList()

#         self.queryd = nn.ModuleList()
#         self.keyd = nn.ModuleList()
#         self.valued = nn.ModuleList()

#         for _ in range(config.transformer["num_heads"]):
#             query = nn.Linear(channel_num[4] // 4, channel_num[4] // 4, bias=False)
#             self.query.append(copy.deepcopy(query))
#         self.key = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
#         self.value = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)

#         self.psi = nn.InstanceNorm2d(self.num_attention_heads)
#         self.psid = nn.InstanceNorm2d(self.num_attention_heads)
#         self.softmax = Softmax(dim=3)
#         self.out = nn.Linear(channel_num[4], channel_num[4], bias=False)
#         self.outd = nn.Linear(channel_num[4], channel_num[4], bias=False)
#         self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
#         self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

#     def forward(self, S, SKV, T, TKV):
#         multi_head_Q_list = []
#         multi_head_K_list = []
#         multi_head_V_list = []

#         multi_head_Qd_list = []
#         multi_head_Kd_list = []
#         multi_head_Vd_list = []

#         Q0, Q1, Q2, Q3 = S.split(S.shape[2] // 4, dim=2)
#         multi_head_Q_list.append(self.query[0](Q0))
#         multi_head_Q_list.append(self.query[1](Q1))
#         multi_head_Q_list.append(self.query[2](Q2))
#         multi_head_Q_list.append(self.query[3](Q3))
#         Q0, Q1, Q2, Q3 = SKV.split(SKV.shape[2] // 4, dim=2)
#         multi_head_K_list.append(self.key(Q0))
#         multi_head_K_list.append(self.key(Q1))
#         multi_head_K_list.append(self.key(Q2))
#         multi_head_K_list.append(self.key(Q3))
#         Q0, Q1, Q2, Q3 = SKV.split(SKV.shape[2] // 4, dim=2)
#         multi_head_V_list.append(self.value(Q0))
#         multi_head_V_list.append(self.value(Q1))
#         multi_head_V_list.append(self.value(Q2))
#         multi_head_V_list.append(self.value(Q3))

#         Q0, Q1, Q2, Q3 = T.split(T.shape[2] // 4, dim=2)
#         multi_head_Qd_list.append(self.query[0](Q0))
#         multi_head_Qd_list.append(self.query[1](Q1))
#         multi_head_Qd_list.append(self.query[2](Q2))
#         multi_head_Qd_list.append(self.query[3](Q3))
#         Q0, Q1, Q2, Q3 = TKV.split(TKV.shape[2] // 4, dim=2)
#         multi_head_Kd_list.append(self.key(Q0))
#         multi_head_Kd_list.append(self.key(Q1))
#         multi_head_Kd_list.append(self.key(Q2))
#         multi_head_Kd_list.append(self.key(Q3))
#         Q0, Q1, Q2, Q3 = TKV.split(TKV.shape[2] // 4, dim=2)
#         multi_head_Vd_list.append(self.value(Q0))
#         multi_head_Vd_list.append(self.value(Q1))
#         multi_head_Vd_list.append(self.value(Q2))
#         multi_head_Vd_list.append(self.value(Q3))

class Attention_org_cross(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_org_cross, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_sizec
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]

        self.query = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(config.transformer["num_heads"]):
            query = nn.Linear(channel_num[4] // 4, channel_num[4] // 4, bias=False)
            self.query.append(copy.deepcopy(query))
        for _ in range(config.transformer["num_heads"]):
            key = nn.Linear(channel_num[4] // 4, channel_num[4] // 4, bias=False)
            self.key.append(copy.deepcopy(key))
        for _ in range(config.transformer["num_heads"]):
            value = nn.Linear(channel_num[4] // 4, channel_num[4] // 4, bias=False)
            self.value.append(copy.deepcopy(value))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.psid = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out = nn.Linear(channel_num[4], channel_num[4], bias=False)
        self.outd = nn.Linear(channel_num[4], channel_num[4], bias=False)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, S, SKV, T, TKV):
        multi_head_Q_list = []
        multi_head_K_list = []
        multi_head_V_list = []

        multi_head_Qd_list = []
        multi_head_Kd_list = []
        multi_head_Vd_list = []

        Q0, Q1, Q2, Q3 = S.split(S.shape[2] // 4, dim=2)
        multi_head_Q_list.append(self.query[0](Q0))
        multi_head_Q_list.append(self.query[1](Q1))
        multi_head_Q_list.append(self.query[2](Q2))
        multi_head_Q_list.append(self.query[3](Q3))
        Q0, Q1, Q2, Q3 = SKV.split(SKV.shape[2] // 4, dim=2)
        multi_head_K_list.append(self.key[0](Q0))
        multi_head_K_list.append(self.key[1](Q1))
        multi_head_K_list.append(self.key[2](Q2))
        multi_head_K_list.append(self.key[3](Q3))
        Q0, Q1, Q2, Q3 = SKV.split(SKV.shape[2] // 4, dim=2)
        multi_head_V_list.append(self.value[0](Q0))
        multi_head_V_list.append(self.value[1](Q1))
        multi_head_V_list.append(self.value[2](Q2))
        multi_head_V_list.append(self.value[3](Q3))

        Q0, Q1, Q2, Q3 = T.split(T.shape[2] // 4, dim=2)
        multi_head_Qd_list.append(self.query[0](Q0))
        multi_head_Qd_list.append(self.query[1](Q1))
        multi_head_Qd_list.append(self.query[2](Q2))
        multi_head_Qd_list.append(self.query[3](Q3))
        Q0, Q1, Q2, Q3 = TKV.split(TKV.shape[2] // 4, dim=2)
        multi_head_Kd_list.append(self.key[0](Q0))
        multi_head_Kd_list.append(self.key[1](Q1))
        multi_head_Kd_list.append(self.key[2](Q2))
        multi_head_Kd_list.append(self.key[3](Q3))
        Q0, Q1, Q2, Q3 = TKV.split(TKV.shape[2] // 4, dim=2)
        multi_head_Vd_list.append(self.value[0](Q0))
        multi_head_Vd_list.append(self.value[1](Q1))
        multi_head_Vd_list.append(self.value[2](Q2))
        multi_head_Vd_list.append(self.value[3](Q3))
        multi_head_Q = torch.stack(multi_head_Q_list, dim=1)
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)
        multi_head_Qd = torch.stack(multi_head_Qd_list, dim=1)
        multi_head_Kd = torch.stack(multi_head_Kd_list, dim=1)
        multi_head_Vd = torch.stack(multi_head_Vd_list, dim=1)

        multi_head_Q = multi_head_Q.transpose(-1, -2)
        multi_head_Qd = multi_head_Qd.transpose(-1, -2)
        multi_head_V = multi_head_V.transpose(-1, -2)
        multi_head_Vd = multi_head_Vd.transpose(-1, -2)

        ########## Cross-Attention ##############
        attention_scores = torch.matmul(multi_head_Q, multi_head_Kd)
        attention_scoresd = torch.matmul(multi_head_Qd, multi_head_K)
        attention_scores = attention_scores / math.sqrt(self.KV_size)
        attention_scoresd = attention_scoresd / math.sqrt(self.KV_size)
        attention_probs = self.softmax(self.psi(attention_scores))
        attention_probsd = self.softmax(self.psid(attention_scoresd))
        attention_probs = self.attn_dropout(attention_probs)
        attention_probsd = self.attn_dropout(attention_probsd)
        context_layer = torch.matmul(attention_probs, multi_head_Vd)
        context_layerd = torch.matmul(attention_probsd, multi_head_V)
        context_layer = context_layer.permute(0, 3, 2, 1).contiguous()
        context_layerd = context_layerd.permute(0, 3, 2, 1).contiguous()
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], context_layer.shape[2] * 4)
        context_layerd = context_layerd.view(context_layerd.shape[0], context_layerd.shape[1], context_layerd.shape[2] * 4)

        O = self.out(context_layer)
        Od = self.outd(context_layerd)
        S2T = self.proj_dropout(O)
        T2S = self.proj_dropout(Od)

        return T2S, S2T

        ########## Self-Attention ##############
        # attention_scores = torch.matmul(multi_head_Q, multi_head_K)
        # attention_scoresd = torch.matmul(multi_head_Qd, multi_head_Kd)
        # attention_scores = attention_scores / math.sqrt(self.KV_size)
        # attention_scoresd = attention_scoresd / math.sqrt(self.KV_size)
        # attention_probs = self.softmax(self.psi(attention_scores))
        # attention_probsd = self.softmax(self.psid(attention_scoresd))
        # attention_probs = self.attn_dropout(attention_probs)
        # attention_probsd = self.attn_dropout(attention_probsd)
        # context_layer = torch.matmul(attention_probs, multi_head_V)
        # context_layerd = torch.matmul(attention_probsd, multi_head_Vd)
        # context_layer = context_layer.permute(0, 3, 2, 1).contiguous()
        # context_layerd = context_layerd.permute(0, 3, 2, 1).contiguous()
        # context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], context_layer.shape[2] * 4)
        # context_layerd = context_layerd.view(context_layerd.shape[0], context_layerd.shape[1], context_layerd.shape[2] * 4)

        # O = self.out(context_layer)
        # Od = self.outd(context_layerd)
        # S = self.proj_dropout(O)
        # T = self.proj_dropout(Od)

        # S = S + T2S
        # T = T + S2T

        # return S, T

class Block_ViT_cross(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT_cross, self).__init__()
        self.attn_normQ = LayerNorm(channel_num[4], eps=1e-6)
        self.attn_normKV = LayerNorm(config.KV_sizec, eps=1e-6)
        self.channel_attn = Attention_org_cross(config, vis, channel_num)

        self.ffn_norm = LayerNorm(channel_num[4], eps=1e-6)
        self.ffn = Mlp(config, channel_num[4], channel_num[4])

    def forward(self, S, T):
        orgS = S; orgT = T
        S = self.attn_normQ(S)
        T = self.attn_normQ(T)
        SKV = self.attn_normKV(S)
        TKV = self.attn_normKV(T)
        S, T = self.channel_attn(S, SKV, T, TKV)
        S = orgS + S; T = orgT + T
        orgS = S; orgT = T
        S = self.ffn_norm(S); T = self.ffn_norm(T)
        S = self.ffn(S); T = self.ffn(T)
        S = orgS + S; T = orgT + T
        return S, T


class Encoder_cross(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder_cross, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(channel_num[4], eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT_cross(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb, embd):
        S = emb; T = embd;
        for layer_block in self.layer:
            S, T = layer_block(S, T)
        S = self.encoder_norm(S)
        T = self.encoder_norm(T)
        return S, T


class ChannelTransformer_share(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512, 1024], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize = patchSize[3]
        self.embeddings = Channel_Embeddings(config, self.patchSize, img_size=img_size,
                                               in_channels=channel_num[4])
        self.encoder = Encoder_cross(config, vis, channel_num)

        self.reconstruct = Reconstruct(channel_num[4], channel_num[4], kernel_size=1,
                                         scale_factor=(self.patchSize, self.patchSize))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, S, T):
        oriS = S; oriT = T
        S = self.embeddings(S)
        T = self.embeddings(T)

        S, T = self.encoder(S, T)  # (B, n_patch, hidden)
        S = self.reconstruct(S) + oriS
        T = self.reconstruct(T) + oriT
        return S, T