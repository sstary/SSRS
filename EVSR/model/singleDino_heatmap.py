import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        xs = self.D_fc1(x.permute(0, 2, 3, 1))
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        xs = xs.permute(0, 3, 1, 2)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    
class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
       
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)
        
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

Dinov3_checkpoint = torch.load("/home/ubuntu22-tmp/xianping/EVSR/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")

class DINOv3(nn.Module):
    def __init__(
        self,
        backbone,
        interaction_indexes=[23],
    ):
        super(DINOv3, self).__init__()
        self.backbone = backbone
        self.interaction_indexes = interaction_indexes
        if "state_dict" in Dinov3_checkpoint:
            state_dict = Dinov3_checkpoint["state_dict"]
        else:
            state_dict = Dinov3_checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # loading weight
        self.backbone.load_state_dict(state_dict, strict=False)
        print("Loading Dinov3 successful!")
        
    def forward(self, x):
        with torch.autocast("cuda", torch.bfloat16):
            all_layers = self.backbone.get_intermediate_layers(
                x, n=self.interaction_indexes
            )
        return all_layers


class SpatialTopKAttention(nn.Module):
    """
    Spatial Top-K Selective Attention (STSA) + Soft-MoE fusion over multiple mask rates.
    - Experts: different top-k masks (ks list)
    - Gating: per-token-per-head MLP producing softmax weights over experts
    - Returns: fused output and optional aux loss (balance loss)
    """

    def __init__(self, dim, num_heads=4, ks=(0.25, 0.5, 0.75), bias=True,
                 noisy_gating=False, noise_std=1.0):
        """
        ks: iterable of fractions (0<k<=1) representing top-k fraction of N. e.g. (0.25, 0.5, 0.75)
            Alternatively you can pass absolute integers, see _parse_ks.
        noisy_gating: whether to add gaussian noise to gate logits during training (encourages exploration)

        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.noisy_gating = noisy_gating
        self.noise_std = noise_std

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # original fixed scalar weights (kept for backward compatibility, but we won't use them)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        # Experts mask specification
        self.ks = self._parse_ks(ks)

        # gating MLP: maps per-head per-token query-vector (head_dim) -> num_experts logits
        head_dim = dim // num_heads
        self.num_experts = len(self.ks)
        # small FFN as gate: a linear layer optionally with hidden nonlinearity (keep tiny)
        self.gate_fc = nn.Sequential(
            nn.Linear(head_dim, max(16, head_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(16, head_dim // 2), self.num_experts)
        )

    def _parse_ks(self, ks):
        """
        Normalize ks input. Accept fractions in (0,1] or integers (>1)
        Return list of fractions if originally fractions, else absolute ints will be computed in forward.
        """
        ks_parsed = []
        for k in ks:
            if isinstance(k, float) and 0 < k <= 1.0:
                ks_parsed.append(k)  # fraction
            elif isinstance(k, int) and k >= 1:
                ks_parsed.append(k)  # absolute - will be handled later
            else:
                raise ValueError("ks must be floats in (0,1] or ints >=1")
        return ks_parsed

    def forward(self, x):
        """
        x: [B, H, W, C] (your original format)
        return_aux_loss: if True, returns (out, aux_loss); otherwise returns out
        """
        b, c, h, w = x.shape
        N = h * w

        # QKV
        qkv = self.qkv_dwconv(self.qkv(x))  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)

        # reshape to heads: [B, head, head_dim, N]
        head_dim = c // self.num_heads
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # normalize along channel dimension for stable dot-product
        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)

        # attention map: [B, head, N, N]
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.temperature  # (Q @ K^T)

        # prepare masks for each expert
        expert_attn_softmax = []
        for k_spec in self.ks:
            # compute integer topk
            if isinstance(k_spec, float) and 0 < k_spec <= 1.0:
                topk = max(1, int(N * k_spec))
            else:
                topk = min(N, int(k_spec))
            # create mask indices and masked softmax
            topk_idx = torch.topk(attn, k=topk, dim=-1).indices  # [B, head, N, topk]
            mask = torch.zeros_like(attn)
            mask.scatter_(-1, topk_idx, 1.0)
            # masked softmax (fill -inf where zero)
            attn_masked = attn.masked_fill(mask == 0, float('-inf')).softmax(dim=-1)
            expert_attn_softmax.append(attn_masked)  # each is [B, head, N, N]

        # compute expert outputs: for each expert, compute out: [B, head, head_dim, N]
        expert_outputs = []
        # we want v in shape [B, head, N, head_dim] for multiplication
        v_t = v.permute(0, 1, 3, 2)  # [B, head, N, head_dim]
        for attn_e in expert_attn_softmax:
            out = (attn_e @ v_t).permute(0, 1, 3, 2)  # back to [B, head, head_dim, N]
            expert_outputs.append(out)

        # stack experts: [E, B, head, head_dim, N]
        expert_stack = torch.stack(expert_outputs, dim=0)

        # ---------- GATING ----------
        # compute gate logits from q per token per head
        # q: [B, head, head_dim, N] -> gate_input: [B, head, N, head_dim]
        gate_input = q.permute(0, 1, 3, 2).contiguous()  # [B, head, N, head_dim]
        B, H, L, D = gate_input.shape  # L == N
        gate_input_flat = gate_input.view(B * H * L, D)  # [B*head*N, head_dim]

        gate_logits = self.gate_fc(gate_input_flat)  # [B*head*N, E]

        if self.noisy_gating and self.training:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.noise_std

        gate_probs = F.softmax(gate_logits, dim=-1)  # [B*head*N, E]
        gate_probs = gate_probs.view(B, H, L, self.num_experts)  # [B, head, N, E]

        # ---------- Combine expert outputs using soft gating ----------
        # expert_stack: [E, B, head, head_dim, N] -> permute to [B, head, E, head_dim, N]
        expert_stack = expert_stack.permute(1, 2, 0, 3, 4)  # [B, head, E, head_dim, N]
        # gate_probs: [B, head, N, E] -> permute to [B, head, E, 1, N] for broadcasting
        gate = gate_probs.permute(0, 1, 3, 2).unsqueeze(3)  # [B, head, E, 1, N]

        weighted = (expert_stack * gate).sum(dim=2)  # sum over E -> [B, head, head_dim, N]

        out = weighted  # [B, head, head_dim, N]

        # reconstruct: [B, head*head_dim, H, W]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()

        self.image_encoder = DINOv3(
            backbone=torch.hub.load(
                "./dinov3",
                'dinov3_vitl16',  # hubconf.py 里定义的函数
                # 'dinov3_vith16plus',  # 1280
                source='local',
                pretrained=False  # 我们用自定义 checkpoint
            ),
            interaction_indexes=[5, 11, 17, 23]
        )

        encoder_channels = (256, 256, 256, 256)
        
        for n, value in self.image_encoder.named_parameters(): 
            if "Adapter" not in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        self.neck1 = nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=1,bias=False,),
            LayerNorm2d(512),
            nn.Conv2d(512,256,kernel_size=3,padding=1,bias=False,),
            LayerNorm2d(256),)
        self.neck2 = nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=1,bias=False,),
            LayerNorm2d(512),
            nn.Conv2d(512,256,kernel_size=3,padding=1,bias=False,),
            LayerNorm2d(256),)
        self.neck3 = nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=1,bias=False,),
            LayerNorm2d(512),
            nn.Conv2d(512,256,kernel_size=3,padding=1,bias=False,),
            LayerNorm2d(256),)
        self.neck4 = nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=1,bias=False,),
            LayerNorm2d(512),
            nn.Conv2d(512,256,kernel_size=3,padding=1,bias=False,),
            LayerNorm2d(256),)

        self.seleAdapter1 = SpatialTopKAttention(dim=256)
        self.seleAdapter2 = SpatialTopKAttention(dim=256)
        self.seleAdapter3 = SpatialTopKAttention(dim=256)
        self.seleAdapter4 = SpatialTopKAttention(dim=256)

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            Norm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def _gradcam_heatmap(self, pred, feature):
        feature_grad = torch.autograd.grad(
            pred,
            feature,
            allow_unused=True,
            retain_graph=True,
        )[0]
        if feature_grad is None:
            return np.zeros(feature.shape[-2:], dtype=np.float32)

        pooled_grads = F.adaptive_avg_pool2d(feature_grad, (1, 1))[0]
        weighted_feature = feature[0] * pooled_grads
        heatmap = weighted_feature.detach().cpu().numpy().mean(axis=0)
        heatmap = np.maximum(heatmap, 0)
        max_value = np.max(heatmap)
        if max_value > 0:
            heatmap = heatmap / max_value
        return heatmap

    def forward(self, x, mode='Train', heatmap_point=(100, 65), heatmap_class=1):
        b, _, h, w = x.size()
        deepx = self.image_encoder(x) # 256*1024
        deepx1 = deepx[0].permute(0, 2, 1).view(b, 1024, 16, 16)
        deepx2 = deepx[1].permute(0, 2, 1).view(b, 1024, 16, 16)
        deepx3 = deepx[2].permute(0, 2, 1).view(b, 1024, 16, 16)
        deepx4 = deepx[3].permute(0, 2, 1).view(b, 1024, 16, 16)
        heatmap_features = [deepx1, deepx2, deepx3, deepx4]

        deepx1 = self.neck1(deepx1)
        deepx1 = self.seleAdapter1(deepx1) + deepx1
        deepx2 = self.neck2(deepx2)
        deepx2 = self.seleAdapter2(deepx2) + deepx2
        deepx3 = self.neck3(deepx3)
        deepx3 = self.seleAdapter3(deepx3) + deepx3
        deepx4 = self.neck4(deepx4)
        deepx4 = self.seleAdapter4(deepx4) + deepx4
        heatmap_features.append(deepx1);heatmap_features.append(deepx2);
        heatmap_features.append(deepx3);heatmap_features.append(deepx4);

        res1 = self.fpn1(deepx1)
        res2 = self.fpn2(deepx2)
        res3 = self.fpn3(deepx3)
        res4 = self.fpn4(deepx4)

        x = self.decoder(res1, res2, res3, res4, h, w)
        if mode == 'Test':
            y_idx, x_idx = heatmap_point
            y_idx = max(0, min(int(y_idx), h - 1))
            x_idx = max(0, min(int(x_idx), w - 1))
            pred = x[:, 4, y_idx, x_idx].sum()
            heatmaps = [self._gradcam_heatmap(pred, feature) for feature in heatmap_features]
            return x, heatmaps
        return x
