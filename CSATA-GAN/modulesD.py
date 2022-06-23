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


class Attention_org_single(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_org_single, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_sizec
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]

        self.query = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(config.transformer["num_heads"]):
            query = nn.Linear(channel_num[4] // 4, channel_num[4] // 4, bias=False)
            key = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            value = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            self.query.append(copy.deepcopy(query))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out = nn.Linear(channel_num[4], channel_num[4], bias=False)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, emb1, emb_all):
        multi_head_Q1_list = []
        multi_head_K_list = []
        multi_head_V_list = []

        if emb1 is not None:
            Q0, Q1, Q2, Q3 = emb1.split(emb1.shape[2] // 4, dim=2)
            multi_head_Q1_list.append(self.query[0](Q0))
            multi_head_Q1_list.append(self.query[1](Q1))
            multi_head_Q1_list.append(self.query[2](Q2))
            multi_head_Q1_list.append(self.query[3](Q3))
        Q0, Q1, Q2, Q3 = emb_all.split(emb_all.shape[2] // 4, dim=2)
        multi_head_K_list.append(self.key[0](Q0))
        multi_head_K_list.append(self.key[0](Q1))
        multi_head_K_list.append(self.key[0](Q2))
        multi_head_K_list.append(self.key[0](Q3))
        Q0, Q1, Q2, Q3 = emb_all.split(emb_all.shape[2] // 4, dim=2)
        multi_head_V_list.append(self.value[0](Q0))
        multi_head_V_list.append(self.value[0](Q1))
        multi_head_V_list.append(self.value[0](Q2))
        multi_head_V_list.append(self.value[0](Q3))

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1)
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)
        multi_head_Q1 = multi_head_Q1.transpose(-1, -2)
        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K)
        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size)
        attention_probs1 = self.softmax(self.psi(attention_scores1))
        if self.vis:
            weights = []
            weights.append(attention_probs1.mean(1))
        else:
            weights = None

        attention_probs1 = self.attn_dropout(attention_probs1)
        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V)
        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous()
        context_layer1 = context_layer1.view(context_layer1.shape[0], context_layer1.shape[1],
                                             context_layer1.shape[2] * 4)

        O1 = self.out(context_layer1)
        O1 = self.proj_dropout(O1)
        return O1, weights

class Block_ViT_single(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT_single, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm = LayerNorm(channel_num[4], eps=1e-6)
        self.attn_normx = LayerNorm(config.KV_sizec, eps=1e-6)
        self.channel_attn = Attention_org_single(config, vis, channel_num)

        self.ffn_norm = LayerNorm(channel_num[4], eps=1e-6)
        self.ffn = Mlp(config, channel_num[4], channel_num[4])

    def forward(self, emb1):
        org1 = emb1
        emb_all = emb1
        cx1 = self.attn_norm(emb1)
        emb_all = self.attn_normx(emb_all)
        cx1, weights = self.channel_attn(cx1, emb_all)
        cx1 = org1 + cx1
        org1 = cx1
        x1 = self.ffn_norm(cx1)
        x1 = self.ffn(x1)
        x1 = x1 + org1

        return x1, weights

class Encoder_single(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder_single, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(channel_num[4], eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT_single(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1):
        attn_weights = []
        for layer_block in self.layer:
            emb1, weights = layer_block(emb1)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm(emb1)
        return emb1, attn_weights

class ChannelTransformer_single(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512, 1024], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize = patchSize[4]
        self.embeddings = Channel_Embeddings(config, self.patchSize, img_size=img_size,
                                               in_channels=channel_num[4])
        self.encoder = Encoder_single(config, vis, channel_num)

        self.reconstruct = Reconstruct(channel_num[4], channel_num[4], kernel_size=1,
                                         scale_factor=(self.patchSize, self.patchSize))

    def forward(self, en1):
        emb1 = self.embeddings(en1)
        encoded1, attn_weights = self.encoder(emb1)  # (B, n_patch, hidden)
        x1 = self.reconstruct(encoded1) if en1 is not None else None
        x1 = x1 + en1 if en1 is not None else None
        return x1, attn_weights

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

        self.queryd = nn.ModuleList()
        self.keyd = nn.ModuleList()
        self.valued = nn.ModuleList()

        for _ in range(config.transformer["num_heads"]):
            query = nn.Linear(channel_num[4] // 4, channel_num[4] // 4, bias=False)
            key = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            value = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            self.query.append(copy.deepcopy(query))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))

            queryd = nn.Linear(channel_num[4] // 4, channel_num[4] // 4, bias=False)
            keyd = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            valued = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            self.queryd.append(copy.deepcopy(queryd))
            self.keyd.append(copy.deepcopy(keyd))
            self.valued.append(copy.deepcopy(valued))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.psid = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out = nn.Linear(channel_num[4], channel_num[4], bias=False)
        self.outd = nn.Linear(channel_num[4], channel_num[4], bias=False)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, emb1, emb_all, embd1, emb_alld):
        multi_head_Q1_list = []
        multi_head_K_list = []
        multi_head_V_list = []

        multi_head_Qd1_list = []
        multi_head_Kd_list = []
        multi_head_Vd_list = []

        if emb1 is not None:
            Q0, Q1, Q2, Q3 = emb1.split(emb1.shape[2] // 4, dim=2)
            multi_head_Q1_list.append(self.query[0](Q0))
            multi_head_Q1_list.append(self.query[1](Q1))
            multi_head_Q1_list.append(self.query[2](Q2))
            multi_head_Q1_list.append(self.query[3](Q3))
        Q0, Q1, Q2, Q3 = emb_all.split(emb_all.shape[2] // 4, dim=2)
        multi_head_K_list.append(self.key[0](Q0))
        multi_head_K_list.append(self.key[0](Q1))
        multi_head_K_list.append(self.key[0](Q2))
        multi_head_K_list.append(self.key[0](Q3))
        Q0, Q1, Q2, Q3 = emb_all.split(emb_all.shape[2] // 4, dim=2)
        multi_head_V_list.append(self.value[0](Q0))
        multi_head_V_list.append(self.value[0](Q1))
        multi_head_V_list.append(self.value[0](Q2))
        multi_head_V_list.append(self.value[0](Q3))

        if embd1 is not None:
            Q0, Q1, Q2, Q3 = embd1.split(embd1.shape[2] // 4, dim=2)
            multi_head_Qd1_list.append(self.queryd[0](Q0))
            multi_head_Qd1_list.append(self.queryd[1](Q1))
            multi_head_Qd1_list.append(self.queryd[2](Q2))
            multi_head_Qd1_list.append(self.queryd[3](Q3))
        Q0, Q1, Q2, Q3 = emb_alld.split(emb_alld.shape[2] // 4, dim=2)
        multi_head_Kd_list.append(self.keyd[0](Q0))
        multi_head_Kd_list.append(self.keyd[0](Q1))
        multi_head_Kd_list.append(self.keyd[0](Q2))
        multi_head_Kd_list.append(self.keyd[0](Q3))
        Q0, Q1, Q2, Q3 = emb_alld.split(emb_alld.shape[2] // 4, dim=2)
        multi_head_Vd_list.append(self.valued[0](Q0))
        multi_head_Vd_list.append(self.valued[0](Q1))
        multi_head_Vd_list.append(self.valued[0](Q2))
        multi_head_Vd_list.append(self.valued[0](Q3))

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1)
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        multi_head_Qd1 = torch.stack(multi_head_Qd1_list, dim=1)
        multi_head_Kd = torch.stack(multi_head_Kd_list, dim=1)
        multi_head_Vd = torch.stack(multi_head_Vd_list, dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2)

        multi_head_Qd1 = multi_head_Qd1.transpose(-1, -2)

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_Kd)

        attention_scoresd1 = torch.matmul(multi_head_Qd1, multi_head_K)

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size)

        attention_scoresd1 = attention_scoresd1 / math.sqrt(self.KV_size)

        attention_probs1 = self.softmax(self.psi(attention_scores1))

        attention_probsd1 = self.softmax(self.psid(attention_scoresd1))
        # print(attention_probs4.size())

        if self.vis:
            weights = []
            weights.append(attention_probs1.mean(1))
        else:
            weights = None

        attention_probs1 = self.attn_dropout(attention_probs1)

        attention_probsd1 = self.attn_dropout(attention_probsd1)

        multi_head_V = multi_head_V.transpose(-1, -2)
        multi_head_Vd = multi_head_Vd.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_Vd)

        context_layerd1 = torch.matmul(attention_probsd1, multi_head_V)

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous()

        context_layerd1 = context_layerd1.permute(0, 3, 2, 1).contiguous()

        context_layer1 = context_layer1.view(context_layer1.shape[0], context_layer1.shape[1],
                                             context_layer1.shape[2] * 4)
        context_layerd1 = context_layerd1.view(context_layerd1.shape[0], context_layerd1.shape[1],
                                               context_layerd1.shape[2] * 4)

        O1 = self.out(context_layer1)
        Od1 = self.outd(context_layerd1)
        O1 = self.proj_dropout(O1)
        Od1 = self.proj_dropout(Od1)
        return O1, Od1, weights


class Block_ViT_cross(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT_cross, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm = LayerNorm(channel_num[4], eps=1e-6)

        self.attn_normd = LayerNorm(channel_num[4], eps=1e-6)

        self.attn_normx = LayerNorm(config.KV_sizec, eps=1e-6)
        self.attn_normy = LayerNorm(config.KV_sizec, eps=1e-6)
        self.channel_attn = Attention_org_cross(config, vis, channel_num)

        self.ffn_norm = LayerNorm(channel_num[4], eps=1e-6)
        self.ffn_normd = LayerNorm(channel_num[4], eps=1e-6)

        self.ffn = Mlp(config, channel_num[4], channel_num[4])
        self.ffnd = Mlp(config, channel_num[4], channel_num[4])

    def forward(self, emb1, embd1):
        org1 = emb1
        orgd1 = embd1
        emb_all = emb1
        emb_alld = embd1
        cx1 = self.attn_norm(emb1)

        cy1 = self.attn_normd(embd1)

        emb_all = self.attn_normx(emb_all)
        emb_alld = self.attn_normy(emb_alld)
        cx1, cy1, weights = self.channel_attn(cx1, emb_all, cy1, emb_alld)
        cx1 = org1 + cx1

        cy1 = orgd1 + cy1

        org1 = cx1

        orgd1 = cy1
        x1 = self.ffn_norm(cx1)
        y1 = self.ffn_normd(cy1)

        x1 = self.ffn(x1)
        y1 = self.ffnd(y1)

        x1 = x1 + org1
        y1 = y1 + orgd1

        return x1, y1, weights


class Encoder_cross(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder_cross, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(channel_num[4], eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT_cross(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, embd1):
        attn_weights = []
        for layer_block in self.layer:
            emb1, embd1, weights = layer_block(emb1, embd1)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm(emb1)
        embd1 = self.encoder_norm(embd1)
        return emb1, embd1, attn_weights


class ChannelTransformer_cross(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512, 1024], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize = patchSize[3]
        self.embeddings = Channel_Embeddings(config, self.patchSize, img_size=img_size,
                                               in_channels=channel_num[4])
        self.embeddingsd = Channel_Embeddings(config, self.patchSize, img_size=img_size,
                                                in_channels=channel_num[4])
        self.encoder = Encoder_cross(config, vis, channel_num)

        self.reconstruct = Reconstruct(channel_num[4], channel_num[4], kernel_size=1,
                                         scale_factor=(self.patchSize, self.patchSize))
        self.reconstruct_d = Reconstruct(channel_num[4], channel_num[4], kernel_size=1,
                                          scale_factor=(self.patchSize, self.patchSize))

    def forward(self, en1, end1):
        emb1 = self.embeddings(en1)

        embd1 = self.embeddingsd(end1)

        encoded1,  encodedd1, attn_weights = self.encoder(
            emb1, embd1)  # (B, n_patch, hidden)
        x1 = self.reconstruct(encoded1) if en1 is not None else None
        y1 = self.reconstruct_d(encodedd1) if en1 is not None else None

        x1 = x1 + en1 if en1 is not None else None
        y1 = y1 + end1 if end1 is not None else None

        return x1, y1, attn_weights
