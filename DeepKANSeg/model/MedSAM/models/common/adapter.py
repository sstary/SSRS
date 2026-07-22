import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from einops import rearrange
from model.kan import KANLinear

# class DW_bn_relu(nn.Module):
#     def __init__(self, dim=768):
#         super(DW_bn_relu, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
#         self.bn = nn.BatchNorm2d(dim)
#         self.relu = nn.ReLU()

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.transpose(1, 2).view(B, C, H, W)
#         x = self.dwconv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = x.flatten(2).transpose(1, 2)

#         return x
    
# class Adapter(nn.Module):
#     def __init__(self, D_features, mlp_ratio=0.1, act_layer=nn.GELU, skip_connect=True):
#         super().__init__()
#         self.skip_connect = skip_connect
#         grid_size=5
#         spline_order=3
#         scale_noise=0.1
#         scale_base=1.0
#         scale_spline=1.0
#         base_activation=torch.nn.SiLU
#         grid_eps=0.02
#         grid_range=[-1, 1]
#         self.mlp_ratio = mlp_ratio

#         D_hidden_features = int(D_features * mlp_ratio)
#         self.act = act_layer()
#         self.D_fc1 = KANLinear(
#                         D_features,
#                         D_hidden_features,
#                         grid_size=grid_size,
#                         spline_order=spline_order,
#                         scale_noise=scale_noise,
#                         scale_base=scale_base,
#                         scale_spline=scale_spline,
#                         base_activation=base_activation,
#                         grid_eps=grid_eps,
#                         grid_range=grid_range,
#                     )
#         self.D_fc2 = KANLinear(
#                         D_hidden_features,
#                         D_features,
#                         grid_size=grid_size,
#                         spline_order=spline_order,
#                         scale_noise=scale_noise,
#                         scale_base=scale_base,
#                         scale_spline=scale_spline,
#                         base_activation=base_activation,
#                         grid_eps=grid_eps,
#                         grid_range=grid_range,
#                     )
#         self.dwconv = DW_bn_relu(D_hidden_features)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         # x is (BT, HW+1, D)
#         B, H, W, C = x.shape
#         xs = x.flatten(1,2)
#         B, N, C = xs.shape

#         xs = self.D_fc1(xs.reshape(B*N,C))
#         xs = xs.reshape(B,N,int(C * self.mlp_ratio)).contiguous()
#         xs = self.dwconv(xs, H, W)
#         xs = self.D_fc2(xs.reshape(B*N,int(C * self.mlp_ratio)))
#         xs = xs.reshape(B,N,C).contiguous()
#         xs = xs.reshape(B, H, W, -1).contiguous()

#         if self.skip_connect:
#             x = x + xs
#         else:
#             x = xs
#         return x
    
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x