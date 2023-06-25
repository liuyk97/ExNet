import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os
from torchvision.models import resnet18
from torchvision.models._utils import IntermediateLayerGetter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def normalization(X):
    size = X.size()
    X_mean, X_std = calc_mean_std(X)
    normalized_feat_X = (X - X_mean.expand(
        size)) / X_std.expand(size)
    return normalized_feat_X


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class SegHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        super(SegHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv2d(inter_channels, channels, 1)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv_out(out)
        return out


class ResConvModule(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResConvModule, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = nn.Conv2d(in_c, out_c, 1, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResDeConvModule(nn.Module):
    def __init__(self, inc1, inc2=0):
        super(ResDeConvModule, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        if inc2 != 0:
            self.up = nn.ConvTranspose2d(inc2, inc2, kernel_size=2, padding=0, stride=2)
        else:
            self.up = nn.ConvTranspose2d(inc1, inc1, kernel_size=2, padding=0, stride=2)
        out_channel = (inc1 + inc2) // 4
        self.conv1 = nn.Conv2d(inc1 + inc2, out_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x = self.up(x2)
            x = torch.cat([x1, x], dim=1)
        else:
            x = self.up(x1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        identity = x

        x = self.conv2(x)
        x = self.bn2(x)

        out = x + identity

        out = self.relu(out)
        return out


class FreBlock(nn.Module):
    def __init__(self, inc):
        super(FreBlock, self).__init__()
        self.convf = nn.Sequential(
            nn.Conv2d(inc * 2, inc * 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc * 2, inc * 2, 1, 1, 0))
        self.convs = nn.Sequential(
            nn.Conv2d(inc, inc, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inc, inc, 3, 1, 1))

    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm='backward')
        x = self.convs(x)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = self.convf(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        return y + x


class apFreBlock(nn.Module):
    def __init__(self, inc):
        super(apFreBlock, self).__init__()
        self.conv_amp = nn.Sequential(
            nn.Conv2d(inc, inc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc, inc, 1, 1, 0), )
        self.conv_pha = nn.Sequential(
            nn.Conv2d(inc, inc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc, inc, 1, 1, 0), )
        self.conv = nn.Sequential(
            nn.Conv2d(inc * 2, inc * 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inc * 2, inc * 2, 3, 1, 1))
        # self.conv_s = nn.Sequential(
        #     nn.Conv2d(inc, inc, 3, 1, 1),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(inc, inc, 3, 1, 1))
        # self.conv_out = nn.Conv2d(inc * 2, inc, 1, 1, 0)

    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x + 1e-5, norm='backward') + 1e-5
        amp = torch.abs(y) + 1e-5
        phase = torch.angle(y) + 1e-5

        amp = self.conv_amp(amp) + 1e-5
        phase = self.conv_pha(phase) + 1e-5

        real = amp * torch.cos(phase) + 1e-5
        imag = amp * torch.sin(phase) + 1e-5

        f = torch.cat([real, imag], dim=1)
        f = self.conv(f) + 1e-5
        real, imag = torch.chunk(f, 2, dim=1)

        y = torch.complex(real, imag) + 1e-5
        y = torch.abs(torch.fft.irfft2(y + 1e-5, s=(H, W), norm='backward') + 1e-5)
        y = torch.nan_to_num(y, nan=0, posinf=1e-5, neginf=1e-5)
        y = y + x
        return y


class StyleNorBlock(nn.Module):
    def __init__(self, num, inc):
        super(StyleNorBlock, self).__init__()
        self.num_nor = num
        self.conv_A = nn.Conv2d(inc, inc, 3, 1, 1)
        self.conv_B = nn.Conv2d(inc, inc, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.cat = nn.Conv2d(inc, inc, 3, 1, 1)

    def forward(self, A, B):
        for i in range(self.num_nor):
            A = self.conv_A(A)
            A = adaptive_instance_normalization(A, B)
            A = self.relu(A)
            B = self.conv_B(B)
            B = adaptive_instance_normalization(B, A)
            B = self.relu(B)

            cat = self.cat(torch.abs(A - B))
            mean, std, var = calc_mean_std(cat)
            mean = mean.expand_as(cat)
            std = std.expand_as(cat)
            # chanel and spatial filter
            # mask = cat < mean - (1 - (cur_iter // self.max_steps)) * std  # 设置一个类似指数移动平均的参数，动态调整阈值
            mask = cat < mean - std  # 设置一个类似指数移动平均的参数，动态调整阈值
            out_A, out_B = torch.zeros_like(A), torch.zeros_like(B)
            out_A[~mask] = A[~mask]
            out_B[~mask] = B[~mask]
            out_A[mask] = B[mask]
            out_B[mask] = A[mask]
            A = out_A
            B = out_B

        return A, B


class ChannelExchange(nn.Module):
    """
    channel exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super(ChannelExchange, self).__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape

        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


class SpatialExchange(nn.Module):
    """
    spatial exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=1 / 2):
        super(SpatialExchange, self).__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2


class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.amp_fuse = nn.Sequential(nn.Conv2d(2 * channels, 2 * channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(2 * channels, 2 * channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2 * channels, 2 * channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(2 * channels, 2 * channels, 1, 1, 0))

    def forward(self, A, B):
        _, _, H, W = A.shape
        A = torch.fft.rfft2(A + 1e-5, norm='backward')
        B = torch.fft.rfft2(B + 1e-5, norm='backward')
        A_amp = torch.abs(A) + 1e-5
        A_pha = torch.angle(A) + 1e-5
        B_amp = torch.abs(B) + 1e-5
        B_pha = torch.angle(B) + 1e-5
        amp_fuse = self.amp_fuse(torch.cat([A_amp, B_amp], 1)) + 1e-5
        pha_fuse = self.pha_fuse(torch.cat([A_pha, B_pha], 1)) + 1e-5
        A_amp, B_amp = torch.chunk(amp_fuse, 2, dim=1)
        A_pha, B_pha = torch.chunk(pha_fuse, 2, dim=1)
        A_real = A_amp * torch.cos(A_pha) + 1e-5
        A_imag = A_amp * torch.sin(A_pha) + 1e-5
        B_real = B_amp * torch.cos(B_pha) + 1e-5
        B_imag = B_amp * torch.sin(B_pha) + 1e-5
        A = torch.complex(A_real, A_imag) + 1e-5
        B = torch.complex(B_real, B_imag) + 1e-5
        A = torch.abs(torch.fft.irfft2(A, s=(H, W), norm='backward') + 1e-5)
        B = torch.abs(torch.fft.irfft2(B, s=(H, W), norm='backward') + 1e-5)
        A = torch.nan_to_num(A, nan=1e-5, posinf=1e-5, neginf=1e-5)
        B = torch.nan_to_num(B, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return A, B


# class Adaptive_Filter_based_Exchange_Module(nn.Module):  # low-pass filter
#     def __init__(self, num_nor, inc, max_steps):
#         super(Adaptive_Filter_based_Exchange_Module, self).__init__()
#         self.num_nor = num_nor
#         self.max_steps = max_steps
#         # self.cat = nn.Conv2d(inc, inc, 3, 1, 1)
#         self.conv_A = nn.Conv2d(inc, inc, 1, 1, 0)
#         self.conv_B = nn.Conv2d(inc, inc, 1, 1, 0)
#         self.relu = nn.ReLU(inplace=False)
#
#     def forward(self, A, B, cur_iter):
#         for i in range(self.num_nor):
#             A = torch.tanh(A)
#             B = torch.tanh(B)
#             cat = torch.abs(A - B)
#             mean, std = calc_mean_std(cat)
#             mean = mean.expand_as(cat)
#             std = std.expand_as(cat)
#
#             # chanel and spatial filter
#             mask = cat < mean - (1 - (2 * cur_iter // self.max_steps)) * std  # 设置一个类似指数移动平均的参数，动态调整阈值
#             out_A, out_B = torch.zeros_like(A), torch.zeros_like(B)
#             out_A[~mask] = A[~mask]
#             out_B[~mask] = B[~mask]
#             out_A[mask] = B[mask]
#             out_B[mask] = A[mask]
#             A = out_A
#             B = out_B
#
#         A = adaptive_instance_normalization(A, B)  # ab ba aba bab
#         B = adaptive_instance_normalization(B, A)
#         # A = self.conv_A(A)
#         # B = self.conv_B(B)
#         return A, B
#
#
# class CD_Net(nn.Module):
#     def __init__(self, max_steps):
#         super(CD_Net, self).__init__()
#
#         self.ResConv1 = ResConvModule(3, 64)
#         self.ResConv2 = ResConvModule(64, 128)
#         self.ResConv3 = ResConvModule(128, 256)
#         self.ResConv4 = ResConvModule(256, 512)
#         return_layers = {}
#         for i in range(4):
#             return_layers['layer{}'.format(4 - i)] = 'layer{}'.format(4 - i)
#         self.resnet18 = IntermediateLayerGetter(resnet18(pretrained=True), return_layers=return_layers)
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, 1, 1, ),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True))
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(128, 128, 3, 1, 1, ),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True))
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(256, 128, 3, 1, 1, ),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True))
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(512, 128, 3, 1, 1, ),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True))
#
#         # self.conv1 = nn.Conv2d(64, 128, 1, 1, 0)
#         # self.conv2 = nn.Conv2d(128, 128, 1, 1, 0)
#         # self.conv3 = nn.Conv2d(256, 128, 1, 1, 0)
#         # self.conv4 = nn.Conv2d(512, 128, 1, 1, 0)
#
#         self.conv_cat = nn.Sequential(
#             nn.Conv2d(128, 128, 3, 1, 1, ),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3, 1, 1, ))
#
#         # self.ResDeConv1 = ResDeConvModule(64, 64)
#         self.ResDeConv4 = ResDeConvModule(128, 128)
#         self.ResDeConv3 = ResDeConvModule(128, 64)
#         self.ResDeConv2 = ResDeConvModule(128, 48)
#         self.ResDeConv1 = ResDeConvModule(44)
#
#         self.SegHead = SegHead(11, 1)
#         self.SegHead1 = SegHead(44, 1)
#         self.SegHead2 = SegHead(48, 1)
#         self.SegHead3 = SegHead(64, 1)
#         self.SegHead4 = SegHead(167, 1)
#
#         # self.AFEM1 = Adaptive_Filter_based_Exchange_Module(num=1, inc=64, max_steps=max_steps, kernel_p=4)
#         self.AFEM2 = Adaptive_Filter_based_Exchange_Module(num_nor=1, inc=128, max_steps=max_steps)
#         self.AFEM3 = Adaptive_Filter_based_Exchange_Module(num_nor=3, inc=256, max_steps=max_steps)
#         self.AFEM4 = Adaptive_Filter_based_Exchange_Module(num_nor=2, inc=512, max_steps=max_steps)
#
#         # self.apFreBlock2 = apFreBlock(64)
#         # self.apFreBlock3 = apFreBlock(128)
#         self.apFreBlock = apFreBlock(128)
#
#         # self.spex = SpatialExchange()
#         # self.chex = ChannelExchange()
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, A, B, cur_iter=0):
#         b, c, h, w = A.shape
#
#         A1 = self.ResConv1(A)
#         B1 = self.ResConv1(B)
#         # A1 = self.apFreBlock1(A1)
#         # B1 = self.apFreBlock1(B1)
#
#         A2 = self.ResConv2(A1)
#         B2 = self.ResConv2(B1)
#         # A2 = self.apFreBlock2(A2)
#         # B2 = self.apFreBlock2(B2)
#         # A2, B2 = self.AFEM2(A2, B2, cur_iter)
#
#         A3 = self.ResConv3(A2)
#         B3 = self.ResConv3(B2)
#         # A3 = self.apFreBlock3(A3)
#         # B3 = self.apFreBlock3(B3)
#         # A3, B3 = self.AFEM3(A3, B3, cur_iter)
#
#         A4 = self.ResConv4(A3)
#         B4 = self.ResConv4(B3)
#         # A4 = self.apFreBlock4(A4)
#         # B4 = self.apFreBlock4(B4)
#
#         # features_A = self.resnet18(A)
#         # features_B = self.resnet18(B)
#         # A1 = features_A['layer1']
#         # A2 = features_A['layer2']
#         # A3 = features_A['layer3']
#         # A4 = features_A['layer4']
#         # B1 = features_B['layer1']
#         # B2 = features_B['layer2']
#         # B3 = features_B['layer3']
#         # B4 = features_B['layer4']
#
#         A4, B4 = self.AFEM4(A4, B4, cur_iter)
#
#         cat1 = self.conv1(torch.abs(A1 - B1))
#         cat2 = self.conv2(torch.abs(A2 - B2))
#         cat3 = self.conv3(torch.abs(A3 - B3))
#         cat4 = self.conv4(torch.abs(A4 - B4))
#         # cat1 = self.conv_cat(self.conv1(torch.abs(A1 - B1)))
#         # cat2 = self.conv_cat(self.conv2(torch.abs(A2 - B2)))
#         # cat3 = self.conv_cat(self.conv3(torch.abs(A3 - B3)))
#         # cat4 = self.conv_cat(self.conv4(torch.abs(A4 - B4)))
#         # cat1 = self.conv_cat(torch.abs(self.conv1(A1) - self.conv1(B1)))
#         # cat2 = self.conv_cat(torch.abs(self.conv2(A2) - self.conv2(B2)))
#         # cat3 = self.conv_cat(torch.abs(self.conv3(A3) - self.conv3(B3)))
#         # cat4 = self.conv_cat(torch.abs(self.conv4(A4) - self.conv4(B4)))
#
#         # A1 = self.conv1(A1)
#         # B1 = self.conv1(B1)
#         # A2 = self.conv2(A2)
#         # B2 = self.conv2(B2)
#         # A3 = self.conv3(A3)
#         # B3 = self.conv3(B3)
#         # A4 = self.conv4(A4)
#         # B4 = self.conv4(B4)
#         #
#         # cat1 = torch.cat([A1, B1], dim=1)  # 64 x 64
#         # cat2 = torch.cat([A2, B2], dim=1)  # 64 x 64
#         # cat3 = torch.cat([A3, B3], dim=1)  # 32 x 32
#         # cat4 = torch.cat([A4, B4], dim=1)  # 16 x 16
#         #
#         # cat1 = self.conv_cat(cat1)
#         # cat2 = self.conv_cat(cat2)
#         # cat3 = self.conv_cat(cat3)
#         # cat4 = self.conv_cat(cat4)
#
#         cat1 = self.apFreBlock(cat1)
#         cat2 = self.apFreBlock(cat2)
#         cat3 = self.apFreBlock(cat3)
#         cat4 = self.apFreBlock(cat4)
#
#         cat3 = self.ResDeConv4(cat3, cat4)
#         cat2 = self.ResDeConv3(cat2, cat3)
#         cat1 = self.ResDeConv2(cat1, cat2)
#         cat = self.ResDeConv1(cat1)
#
#         cat3 = F.interpolate(cat3, size=[h, w], mode='bilinear', align_corners=True)
#         cat2 = F.interpolate(cat2, size=[h, w], mode='bilinear', align_corners=True)
#         cat1 = F.interpolate(cat1, size=[h, w], mode='bilinear', align_corners=True)
#         cat = F.interpolate(cat, size=[h, w], mode='bilinear', align_corners=True)
#         cat_all = torch.cat([cat, cat1, cat2, cat3], dim=1)
#
#         result = torch.sigmoid(self.SegHead(cat))
#         result1 = torch.sigmoid(self.SegHead1(cat1))
#         result2 = torch.sigmoid(self.SegHead2(cat2))
#         result3 = torch.sigmoid(self.SegHead3(cat3))
#         result4 = torch.sigmoid(self.SegHead4(cat_all))
#
#         return result, result1, result2, result3, result4


class Adaptive_Filter_based_Exchange_Module(nn.Module):  # low-pass filter
    def __init__(self, num, inc, max_steps):
        super(Adaptive_Filter_based_Exchange_Module, self).__init__()
        self.num_nor = num
        self.max_steps = max_steps
        self.conv_A = nn.Conv2d(inc, inc, 3, 1, 1)
        self.conv_B = nn.Conv2d(inc, inc, 3, 1, 1)
        self.relu = nn.ReLU(inplace=False)
        # self.cat = nn.Conv2d(inc, inc, 3, 1, 1)
        # self.conv_A_o = nn.Conv2d(inc, inc, 1, 1, 0)
        # self.conv_B_o = nn.Conv2d(inc, inc, 1, 1, 0)

    def forward(self, A, B, cur_iter):
        for i in range(self.num_nor):
            A = self.conv_A(A)
            A = adaptive_instance_normalization(A, B)
            A = self.relu(A)

            B = self.conv_B(B)
            B = adaptive_instance_normalization(B, A)
            B = self.relu(B)

            cat = torch.abs(A - B)
            mean, std = calc_mean_std(cat)
            mean = mean.expand_as(cat)
            std = std.expand_as(cat)
            th = mean + (1 - (2 * cur_iter // self.max_steps)) * std
            # th= mean
            # th = mean - std
            mask = cat < th  # 设置一个类似指数移动平均的参数，动态调整阈值
            out_A, out_B = torch.zeros_like(A), torch.zeros_like(B)
            out_A[~mask] = A[~mask]
            out_B[~mask] = B[~mask]
            out_A[mask] = B[mask]
            out_B[mask] = A[mask]
            A = out_A
            B = out_B
        return A, B


class CD_Net(nn.Module):
    def __init__(self, max_steps=400000):
        super(CD_Net, self).__init__()

        self.ResConv1 = ResConvModule(3, 64)
        self.ResConv2 = ResConvModule(64, 128)
        self.ResConv3 = ResConvModule(128, 256)

        self.ResDeConv1 = ResDeConvModule(256, 256)
        self.ResDeConv2 = ResDeConvModule(256, 128)
        self.ResDeConv3 = ResDeConvModule(96)

        # self.AFEM1 = Adaptive_Filter_based_Exchange_Module(num=1, inc=64, max_steps=max_steps)
        # self.AFEM2 = Adaptive_Filter_based_Exchange_Module(num=1, inc=128, max_steps=max_steps)
        self.AFEM3 = Adaptive_Filter_based_Exchange_Module(num=1, inc=256, max_steps=max_steps)
        # self.apFreBlock1 = apFreBlock(64)
        # self.apFreBlock2 = apFreBlock(128)
        self.apFreBlock3 = apFreBlock(256)

        # self.channelex = ChannelExchange()
        # self.spatialex = SpatialExchange()
        #
        #

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.conv_cat = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, ))
        self.relu = nn.ReLU(inplace=True)
        self.SegHead3 = SegHead(128, 1)
        self.SegHead2 = SegHead(96, 1)
        self.SegHead1 = SegHead(24, 1)

    def forward(self, A, B, cur_iter=0):
        b, c, h, w = A.shape

        A1 = self.ResConv1(A)
        B1 = self.ResConv1(B)

        A2 = self.ResConv2(A1)
        B2 = self.ResConv2(B1)
        # A2, B2 = self.spatialex(A2, B2)

        A3 = self.ResConv3(A2)
        B3 = self.ResConv3(B2)
        # A3, B3 = self.channelex(A3, B3)

        A3, B3 = self.AFEM3(A3, B3, cur_iter)

        A1 = self.conv1(A1)
        B1 = self.conv1(B1)
        A2 = self.conv2(A2)
        B2 = self.conv2(B2)
        A3 = self.conv3(A3)
        B3 = self.conv3(B3)

        cat1 = torch.cat([A1, B1], dim=1)  # 128 x 128
        cat2 = torch.cat([A2, B2], dim=1)  # 64 x 64
        cat3 = torch.cat([A3, B3], dim=1)  # 32 x 32
        cat1 = self.conv_cat(cat1)
        cat2 = self.conv_cat(cat2)
        cat3 = self.conv_cat(cat3)

        # cat1 = self.apFreBlock3(cat1)
        # cat2 = self.apFreBlock3(cat2)
        # cat3 = self.apFreBlock3(cat3)

        cat2 = self.ResDeConv1(cat2, cat3)
        cat1 = self.ResDeConv2(cat1, cat2)
        cat = self.ResDeConv3(cat1)

        result3 = torch.sigmoid(F.interpolate(self.SegHead3(cat2), size=[h, w], mode='bicubic', align_corners=True))
        result2 = torch.sigmoid(F.interpolate(self.SegHead2(cat1), size=[h, w], mode='bicubic', align_corners=True))
        result1 = torch.sigmoid(self.SegHead1(cat))
        return result1, result2, result3
